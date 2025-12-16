from pathlib import Path
import argparse
import sys
import numpy as np

from .pipeline import GameProcessor
from .video_io import VideoLoader
from .detection import run_detection_on_segment
from .tracking import track_segment
from .teams import assign_teams
from .homography import build_homographies
from .viz import annotate_frame, save_debug_frame, render_video


def test_video(video_path: str) -> None:
    """
    Simple command to verify video loading.
    """
    print(f"Testing video load: {video_path}")
    try:
        with VideoLoader(video_path) as loader:
            info = loader.get_info()
            print(f"Success!")
            print(f"  Dimensions: {info.width}x{info.height}")
            print(f"  FPS: {info.fps}")
            print(f"  Total Frames: {info.total_frames}")
            print(f"  Duration: {info.duration_seconds:.2f}s")
            
            # Optional: read first frame
            print("  Reading first frame...", end=" ")
            for idx, frame in loader:
                print(f"OK (Shape: {frame.shape})")
                break
    except Exception as e:
        print(f"\nError loading video: {e}")
        sys.exit(1)


def test_detection(video_path: str, out_dir: str, frames: int = 1) -> None:
    """
    Run detection on a few frames and save image results.
    """
    print(f"Testing detection on {video_path} (limit {frames} frames)")
    out_path = Path(out_dir)
    
    try:
        with VideoLoader(video_path) as loader:
            # Collect first N frames
            segment = []
            count = 0
            for idx, frame in loader:
                if count >= frames:
                    break
                segment.append((idx, frame))
                count += 1
            
            if not segment:
                print("No frames read from video.")
                return

            # Run detection
            print("Running detection...")
            detections = run_detection_on_segment(segment)
            
            # Group by frame
            dets_by_frame = {}
            for d in detections:
                dets_by_frame.setdefault(d.frame_idx, []).append(d)
            
            # Visualize
            for idx, frame in segment:
                dets = dets_by_frame.get(idx, [])
                print(f"Frame {idx}: {len(dets)} detections")
                annotated = annotate_frame(frame, dets)
                save_debug_frame(annotated, out_path, prefix=f"det_frame_{idx}")

    except Exception as e:
        print(f"\nError during detection test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_tracking(video_path: str, out_dir: str, frames: int = 30) -> None:
    """
    Run tracking on a short video segment and render a video.
    INCLUDES TEAM ASSIGNMENT + COURT MAP.
    """
    print(f"Testing tracking + teams + map on {video_path} (limit {frames} frames)")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_out = out_path / "tracking_full_test.mp4"
    
    try:
        with VideoLoader(video_path) as loader:
            info = loader.get_info()
            
            # Collect frames
            segment = []
            frames_dict = {}
            count = 0
            for idx, frame in loader:
                if count >= frames:
                    break
                segment.append((idx, frame))
                frames_dict[idx] = frame
                count += 1
            
            if not segment:
                print("No frames read.")
                return
            
            # 1. Detection
            print("Running detection...")
            detections = run_detection_on_segment(segment)
            
            # 2. Tracking
            print("Running tracking...")
            tracks = track_segment(detections, fps=info.fps)
            
            # 3. Team Assignment
            print("Assigning teams...")
            team_map = assign_teams(tracks, frames_dict)
            print(f"Assigned teams for {len(team_map)} tracks.")

            # 4. Homography
            print("Computing homographies...")
            # Note: passing just the video path might be cleaner if we implemented that way,
            # but our build_homographies takes frames_dict.
            homography_map = build_homographies(video_path, frames_dict)

            # Group tracks by frame
            tracks_by_frame = {}
            for t in tracks:
                tracks_by_frame.setdefault(t.frame_idx, []).append(t)
            
            # 5. Render
            print("Rendering output video...")
            render_video(
                segment, 
                tracks_by_frame, 
                video_out, 
                fps=info.fps, 
                team_map=team_map,
                homography_map=homography_map
            )
            
    except Exception as e:
        print(f"\nError during tracking test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a basketball game into shots + box score.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'process-game' command
    process_parser = subparsers.add_parser("process-game", help="Run full pipeline on a video")
    process_parser.add_argument("--video", required=True, help="Path to game video")
    process_parser.add_argument("--out", required=True, help="Output directory")

    # 'test-video' command
    test_parser = subparsers.add_parser("test-video", help="Verify video file can be read")
    test_parser.add_argument("--video", required=True, help="Path to game video")

    # 'test-detection' command
    det_parser = subparsers.add_parser("test-detection", help="Run detection on a snippet")
    det_parser.add_argument("--video", required=True, help="Path to game video")
    det_parser.add_argument("--out", required=True, help="Output directory for images")
    det_parser.add_argument("--frames", type=int, default=1, help="Number of frames to process")

    # 'test-tracking' command
    track_parser = subparsers.add_parser("test-tracking", help="Run tracking on a snippet and save video")
    track_parser.add_argument("--video", required=True, help="Path to game video")
    track_parser.add_argument("--out", required=True, help="Output directory for video")
    track_parser.add_argument("--frames", type=int, default=30, help="Number of frames to process")

    args = parser.parse_args()

    if args.command == "process-game":
        gp = GameProcessor(Path(args.video), Path(args.out))
        gp.run()
    
    elif args.command == "test-video":
        test_video(args.video)
        
    elif args.command == "test-detection":
        test_detection(args.video, args.out, args.frames)
    
    elif args.command == "test-tracking":
        test_tracking(args.video, args.out, args.frames)
