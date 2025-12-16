from pathlib import Path
from typing import List, Dict
import numpy as np

from .video_io import VideoLoader
from . import detection, tracking, teams, events, homography, stats
from .models import ShotEvent


class GameProcessor:
    def __init__(self, video_path: Path, out_dir: Path):
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        High-level orchestration:
        1. Load video frames
        2. Detect + track objects
        3. Identify players (teams)
        4. Detect shot events
        5. Map to court coordinates
        6. Export shots + box score
        """
        # 1. Load video and collect frames
        loader = VideoLoader(self.video_path)
        video_info = loader.get_info()
        
        print(f"Processing video: {self.video_path}")
        print(f"Resolution: {video_info.width}x{video_info.height}, FPS: {video_info.fps:.2f}, Duration: {video_info.duration_seconds:.1f}s")
        
        # Collect all frames into a dict for later use (teams/homography need random access)
        # For long videos, you'd want to process in chunks or stream more carefully
        frames_dict: Dict[int, np.ndarray] = {}
        frame_iter = []  # Also build list for detection pass
        
        print("Loading frames...")
        for frame_idx, frame in loader:
            frames_dict[frame_idx] = frame
            frame_iter.append((frame_idx, frame))
        loader.close()
        print(f"Loaded {len(frames_dict)} frames.")

        # 2. Detection + tracking
        dets = detection.run_detection_on_segment(iter(frame_iter))
        tracks = tracking.track_segment(dets, fps=video_info.fps)
        print(f"Tracking complete. {len(tracks)} tracked detections.")

        # 3. Team assignment (skip jersey numbers for now - not implemented)
        team_map = teams.assign_teams(tracks, frames_dict)
        number_map: Dict[int, str] = {}  # TODO: implement numbers.py
        
        # 4. Shot events
        segment_meta = {
            "period": 1,
            "start_clock_s": video_info.duration_seconds,
            "end_clock_s": 0.0,
        }
        shot_events = events.detect_shot_events(
            dets=dets,
            tracks=tracks,
            number_map=number_map,
            team_map=team_map,
            segment_meta=segment_meta,
        )
        print(f"Detected {len(shot_events)} shot events.")

        # 5. Homography - add court coordinates to shots
        H_lookup = homography.build_homographies(str(self.video_path), frames_dict)

        all_shots_with_coords: List[ShotEvent] = []
        for s in shot_events:
            transformer = homography.pick_homography_for_frame(H_lookup, s.video_frame_idx)
            if transformer is not None:
                # Create a dummy detection-like object for coordinate transform
                from .detection import Detection
                shot_det = Detection(
                    frame_idx=s.video_frame_idx,
                    cls="shot",
                    class_id=-1,
                    xyxy=(0, 0, 0, 0),  # Would need actual shot location
                    score=1.0
                )
                x_ft, y_ft = homography.add_court_coordinates(transformer, shot_det)
                s.x_ft = x_ft
                s.y_ft = y_ft
                # Compute distance to hoop (hoop at ~4.25ft from baseline, center court width)
                if x_ft is not None and y_ft is not None:
                    # NBA court: hoop at x=5.25ft from baseline, y=25ft (center)
                    hoop_x, hoop_y = 5.25, 25.0
                    s.distance_ft = np.sqrt((x_ft - hoop_x)**2 + (y_ft - hoop_y)**2)
            all_shots_with_coords.append(s)

        # 6. Exports
        shots_csv = stats.export_shots_csv(all_shots_with_coords, self.out_dir)
        print(f"Exported shots to {shots_csv}")
        
        box_csv = stats.export_box_score_csv(shots_csv, self.out_dir)
        print(f"Exported box score to {box_csv}")
