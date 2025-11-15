from pathlib import Path
from typing import List

from . import video_io, detection, tracking, numbers, teams, events, homography, stats
from .models import ShotEvent


class GameProcessor:
    def __init__(self, video_path: Path, out_dir: Path):
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """
        High-level orchestration:
        1. Segment video
        2. Detect + track objects
        3. Identify players (numbers + teams)
        4. Detect shot events
        5. Map to court
        6. Export shots + box score
        """
        # 1. Load and segment video
        vid = video_io.open_video(self.video_path)
        segments = video_io.segment_video(vid)

        all_shots: List[ShotEvent] = []

        for segment in segments:
            frames = video_io.iter_segment_frames(vid, segment)

            # 2. Detection + tracking
            dets = detection.run_detection_on_segment(frames)
            tracks = tracking.track_segment(dets)

            # 3. Player identity (numbers + teams)
            number_map = numbers.assign_jersey_numbers(tracks, frames)
            team_map = teams.assign_teams(tracks)

            # 4. Shot events (per segment)
            segment_shots = events.detect_shot_events(
                dets=dets,
                tracks=tracks,
                number_map=number_map,
                team_map=team_map,
                segment_meta=segment,
            )
            all_shots.extend(segment_shots)

        # 5. Homography (could be per-camera; use metadata if you have it)
        H_lookup = homography.build_homographies(self.video_path)

        all_shots_with_coords: List[ShotEvent] = []
        for s in all_shots:
            H = homography.pick_homography_for_frame(
                H_lookup, s.video_frame_idx)
            if H is not None:
                x_ft, y_ft, distance_ft = homography.add_court_coordinates(
                    H, s)
                s.x_ft = x_ft
                s.y_ft = y_ft
                s.distance_ft = distance_ft
            all_shots_with_coords.append(s)

        # 6. Exports
        shots_csv = stats.export_shots_csv(all_shots_with_coords, self.out_dir)
        stats.export_box_score_csv(shots_csv, self.out_dir)
