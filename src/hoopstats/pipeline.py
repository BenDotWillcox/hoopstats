from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .video_io import VideoLoader
from . import detection, tracking, teams, events, homography, stats
from .detection import Detection
from .models import ShotEvent

# Frames around each shot release where we try to compute a homography.
SHOT_HOMOGRAPHY_OFFSETS = (0, -5, 5, -10, 10, -20, 20)


class GameProcessor:
    def __init__(self, video_path: Path, out_dir: Path,
                 max_frames: Optional[int] = None,
                 start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 segments: Optional[List[Tuple[int, int]]] = None,
                 enable_number_ocr: bool = True):
        """
        Args:
            max_frames: process frames [0, max_frames) (legacy convenience).
            start_frame/end_frame: process a single explicit frame range.
            segments: list of (start, end) ranges to process and concatenate.
                Used for uncut broadcasts so replays/alt-angles are skipped and
                memory stays bounded (only one segment is held at a time).
                Takes precedence over start/end/max_frames.
            enable_number_ocr: run jersey OCR at shot frames.
        """
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.segments = segments
        self.enable_number_ocr = enable_number_ocr

    def _resolve_ranges(self, total_frames: int) -> List[Tuple[int, int]]:
        """Turn the configured scope into a list of (start, end) frame ranges."""
        if self.segments:
            return [(s, min(e, total_frames)) for s, e in self.segments
                    if s < total_frames]
        if self.end_frame is not None or self.start_frame:
            end = self.end_frame if self.end_frame is not None else total_frames
            return [(self.start_frame, min(end, total_frames))]
        if self.max_frames is not None:
            return [(0, min(self.max_frames, total_frames))]
        return [(0, total_frames)]

    def run(self) -> None:
        """
        Orchestrate the full pipeline over the configured frame ranges and
        export a single shots.csv / box_score.csv keyed by absolute frame index.
        """
        loader = VideoLoader(self.video_path)
        video_info = loader.get_info()
        loader.close()

        print(f"Processing video: {self.video_path}")
        print(f"Resolution: {video_info.width}x{video_info.height}, "
              f"FPS: {video_info.fps:.2f}, Duration: {video_info.duration_seconds:.1f}s")

        ranges = sorted(self._resolve_ranges(video_info.total_frames))
        print(f"Processing {len(ranges)} range(s): "
              f"{', '.join(f'[{s}, {e})' for s, e in ranges)}")

        all_shots = self._process_ranges(ranges, video_info.fps)
        all_shots.sort(key=lambda s: s.video_frame_idx)
        located = sum(1 for s in all_shots if s.x_ft is not None)
        makes = sum(1 for s in all_shots if s.result == "make")
        print(f"\nTotal: {len(all_shots)} shots ({makes} makes); "
              f"court coordinates resolved for {located}.")

        shots_csv = stats.export_shots_csv(all_shots, self.out_dir)
        print(f"Exported shots to {shots_csv}")
        box_csv = stats.export_box_score_csv(shots_csv, self.out_dir)
        print(f"Exported box score to {box_csv}")

    def _process_ranges(self, ranges: List[Tuple[int, int]], fps: float) -> List[ShotEvent]:
        """
        Stream the video once, dispatching frames to their range and processing
        each range's frames when it completes.

        A single forward pass (no per-range seeking) decodes every frame at most
        once and holds only the current range's frames in memory, which keeps a
        full-length broadcast tractable when split into reasonably sized
        segments.
        """
        if not ranges:
            return []

        all_shots: List[ShotEvent] = []
        loader = VideoLoader(self.video_path)
        cur = 0
        buf: Dict[int, np.ndarray] = {}
        last_end = ranges[-1][1]

        try:
            for frame_idx, frame in loader:
                if frame_idx >= last_end:
                    break
                # Finish any ranges that end at/before this frame.
                while cur < len(ranges) and frame_idx >= ranges[cur][1]:
                    if buf:
                        all_shots.extend(self._process_frames(buf, cur, len(ranges), fps))
                        buf = {}
                    cur += 1
                if cur >= len(ranges):
                    break
                start, end = ranges[cur]
                if start <= frame_idx < end:
                    buf[frame_idx] = frame
        finally:
            loader.close()

        if buf and cur < len(ranges):
            all_shots.extend(self._process_frames(buf, cur, len(ranges), fps))

        return all_shots

    def _process_frames(
        self, frames_dict: Dict[int, np.ndarray], range_idx: int,
        n_ranges: int, fps: float,
    ) -> List[ShotEvent]:
        """
        Detect → track → teams → events → homography for one range's frames.

        Returns located ShotEvents with absolute frame indices. Team ids are
        assigned per range (the appearance classifier is fit on this range
        only), so they are locally consistent but not globally aligned across
        ranges — evaluation handles team identity up to a permutation.
        """
        keys = sorted(frames_dict)
        print(f"\n=== Range {range_idx + 1}/{n_ranges}: "
              f"frames [{keys[0]}, {keys[-1]}] ({len(keys)} frames) ===")
        frame_iter = [(k, frames_dict[k]) for k in keys]

        # Detection + tracking
        dets = detection.run_detection_on_segment(iter(frame_iter))
        tracks = tracking.track_segment(dets, fps=fps)
        print(f"Tracking complete. {len(tracks)} tracked detections.")

        # Team assignment (per-range)
        team_map = teams.assign_teams(tracks, frames_dict)
        number_map: Dict[int, str] = {}

        # Shot events. Clock is rough: assume the range plays out from its start.
        range_duration_s = len(frames_dict) / fps if fps else 0.0
        segment_meta = {
            "period": 1,
            "start_clock_s": range_duration_s,
            "end_clock_s": 0.0,
        }
        shot_events = events.detect_shot_events(
            dets=dets, tracks=tracks, number_map=number_map, team_map=team_map,
            segment_meta=segment_meta, fps=fps,
        )
        print(f"Detected {len(shot_events)} shot events "
              f"({sum(1 for s in shot_events if s.result == 'make')} makes).")

        if self.enable_number_ocr and shot_events:
            self._attribute_jersey_numbers(shot_events, dets, frames_dict)

        # Homography only near shot frames within this range.
        shot_frames = set()
        for s in shot_events:
            for off in SHOT_HOMOGRAPHY_OFFSETS:
                f = s.video_frame_idx + off
                if f in frames_dict:
                    shot_frames.add(f)
        shot_frames_dict = {f: frames_dict[f] for f in sorted(shot_frames)}
        H_lookup = homography.build_homographies(str(self.video_path), shot_frames_dict)

        for s in shot_events:
            transformer = homography.pick_homography_for_frame(H_lookup, s.video_frame_idx)
            if transformer is not None and s.shooter_xyxy is not None:
                shot_det = Detection(
                    frame_idx=s.video_frame_idx, cls="shot", class_id=-1,
                    xyxy=tuple(s.shooter_xyxy), score=1.0,
                )
                x_ft, y_ft = homography.add_court_coordinates(transformer, shot_det)
                if x_ft is not None and y_ft is not None:
                    s.x_ft = round(x_ft, 1)
                    s.y_ft = round(y_ft, 1)
                    shot_type, dist = events.classify_shot_type(x_ft, y_ft)
                    s.shot_type = shot_type
                    s.distance_ft = round(dist, 1)

        return shot_events

    def _attribute_jersey_numbers(
        self,
        shot_events: List[ShotEvent],
        dets: List[Detection],
        frames_dict: Dict[int, np.ndarray],
        search_window: int = 15,
    ) -> None:
        """
        Best-effort jersey number OCR for shooters.

        For each shot, look for a `number` detection whose center lies inside
        the shooter's box within a few frames of the release, crop it, and run
        the hosted OCR model. Failures leave shooter_number as None.
        """
        try:
            from .numbers import NumberRecognizer
            recognizer = NumberRecognizer()
        except Exception as e:
            print(f"Warning: jersey OCR unavailable ({e}); skipping number attribution.")
            return

        for s in shot_events:
            if s.shooter_number or s.shooter_xyxy is None:
                continue
            x1, y1, x2, y2 = s.shooter_xyxy

            candidates = []
            for d in dets:
                if d.class_id != detection.NUMBER_CLASS_ID:
                    continue
                if abs(d.frame_idx - s.video_frame_idx) > search_window:
                    continue
                cx = (d.xyxy[0] + d.xyxy[2]) / 2
                cy = (d.xyxy[1] + d.xyxy[3]) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    candidates.append(d)
            if not candidates:
                continue

            best = max(candidates, key=lambda d: d.score)
            frame = frames_dict.get(best.frame_idx)
            if frame is None:
                continue
            bx1, by1, bx2, by2 = best.xyxy
            pad = 10
            h, w = frame.shape[:2]
            crop = frame[max(0, by1 - pad):min(h, by2 + pad),
                         max(0, bx1 - pad):min(w, bx2 + pad)]
            if crop.size == 0:
                continue

            try:
                numbers = recognizer.recognize_crops([crop])
            except Exception as e:
                print(f"Warning: OCR failed at frame {best.frame_idx}: {e}")
                continue
            if numbers and numbers[0]:
                s.shooter_number = numbers[0]
                if s.offense_team_id is not None:
                    s.shooter_global_id = f"{s.offense_team_id}#{numbers[0]}"
