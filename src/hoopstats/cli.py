from pathlib import Path
import argparse
import sys
import time
import traceback
import numpy as np

from .pipeline import GameProcessor
from .video_io import VideoLoader
from .detection import (
    run_detection_on_segment, load_model_if_needed,
    PLAYER_DETECTION_MODEL_CONFIDENCE, PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    CLASS_NAMES, PLAYER_CLASS_IDS, NUMBER_CLASS_ID
)
from .tracking import track_segment
from .teams import assign_teams
from .homography import (
    build_homographies, load_keypoint_model, _keypoint_model,
    KEYPOINT_DETECTION_MODEL_ID, KEYPOINT_DETECTION_MODEL_CONFIDENCE,
    KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE
)
from .viz import annotate_frame, save_debug_frame, render_video
from .numbers import NumberRecognizer, NumberValidator, extract_number_detections
from .sam2_tracking import load_sam2_predictor, SAM2Tracker
from .possessions import (
    load_trajectories, segment_possessions, normalize_all_possessions,
    save_possessions, load_possessions
)
from .clustering import (
    compute_distance_matrix, cluster_possessions, get_cluster_summary
)
from .play_viz import (
    draw_possession_paths, draw_cluster_summary, save_cluster_visualizations,
    render_possession_video, get_video_clip_info
)


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


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}m {remainder:.0f}s"


def _run_demo_stage(name: str, description: str, action, expected_outputs: list[Path]) -> dict:
    """
    Run a demo stage and return report metadata.

    Existing CLI helpers call sys.exit(1) on failure. Catch that here so a
    partial demo still leaves a useful report and any successful artifacts.
    """
    print(f"\n=== Demo stage: {name} ===")
    started = time.perf_counter()
    status = "ok"
    error = ""

    try:
        action()
    except SystemExit as exc:
        code = exc.code if exc.code is not None else 0
        if code != 0:
            status = "failed"
            error = f"exited with code {code}"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    duration = time.perf_counter() - started
    outputs = [p for p in expected_outputs if p.exists()]

    if status == "ok":
        print(f"Stage complete: {name} ({_format_duration(duration)})")
    else:
        print(f"Stage failed: {name} ({error})")

    return {
        "name": name,
        "description": description,
        "status": status,
        "error": error,
        "duration_s": duration,
        "outputs": outputs,
    }


def _write_demo_report(
    report_path: Path,
    video_path: Path,
    out_dir: Path,
    frame_num: int,
    metadata: dict,
    stages: list[dict],
) -> None:
    lines = [
        "# HoopStats Demo Report",
        "",
        "## Input",
        "",
        f"- Video: `{video_path}`",
        f"- Output directory: `{out_dir}`",
        f"- Representative frame: `{frame_num}`",
        "",
        "## Video Metadata",
        "",
    ]

    if metadata:
        lines.extend([
            f"- Resolution: `{metadata['width']}x{metadata['height']}`",
            f"- FPS: `{metadata['fps']:.2f}`",
            f"- Total frames: `{metadata['total_frames']}`",
            f"- Duration: `{metadata['duration_seconds']:.2f}s`",
        ])
    else:
        lines.append("- Metadata unavailable.")

    lines.extend([
        "",
        "## Stages",
        "",
        "| Stage | Status | Duration | Outputs |",
        "| --- | --- | ---: | --- |",
    ])

    for stage in stages:
        if stage["outputs"]:
            output_text = "<br>".join(f"`{p.relative_to(out_dir)}`" for p in stage["outputs"])
        elif stage["error"]:
            output_text = stage["error"]
        else:
            output_text = ""
        lines.append(
            f"| {stage['name']} | {stage['status']} | "
            f"{_format_duration(stage['duration_s'])} | {output_text} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This demo records the current visual pipeline artifacts only.",
        "- Trajectory export, possession segmentation, and play clustering are planned next MVP stages.",
        "- Shot event detection and box score generation are not part of this reproducible demo yet.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def demo(
    video_path: str,
    out_dir: str,
    frame_num: int = 0,
    train_stride: int = 30,
    skip_court_map: bool = False,
    skip_court_video: bool = False,
    include_detection_video: bool = False,
    debug_video: bool = True,
    max_frames: int | None = None,
) -> None:
    """
    Run the reproducible portfolio demo against a single sample video.
    """
    source = Path(video_path)
    out_path = Path(out_dir)
    frames_dir = out_path / "frames"
    videos_dir = out_path / "videos"

    frames_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running demo on: {source}")
    print(f"Writing outputs to: {out_path}")

    metadata = {}
    stages = []

    def collect_metadata():
        nonlocal metadata
        with VideoLoader(source) as loader:
            info = loader.get_info()
            metadata = {
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "total_frames": info.total_frames,
                "duration_seconds": info.duration_seconds,
            }
            print(f"  {info.width}x{info.height}, {info.fps:.2f} FPS, {info.total_frames} frames")

    stages.append(_run_demo_stage(
        "video metadata",
        "Read basic video dimensions, frame count, FPS, and duration.",
        collect_metadata,
        [],
    ))

    stages.append(_run_demo_stage(
        "single-frame detection",
        "Detect players, jersey numbers, ball/rim events, and refs on one frame.",
        lambda: detect_frame(str(source), str(frames_dir), frame_num, "all"),
        [frames_dir / "single_frame_detection" / f"detect_frame_{frame_num}.jpg"],
    ))

    stages.append(_run_demo_stage(
        "court keypoints",
        "Detect court landmarks used for homography.",
        lambda: detect_keypoints(str(source), str(frames_dir), frame_num),
        [frames_dir / "keypoint_detection" / f"keypoints_frame_{frame_num}.jpg"],
    ))

    if not skip_court_map:
        stages.append(_run_demo_stage(
            "single-frame court map",
            "Project detected players from one broadcast frame onto a top-down court.",
            lambda: map_court(str(source), str(frames_dir), frame_num, train_stride=train_stride),
            [
                frames_dir / "court_map" / f"teams_frame_{frame_num}.jpg",
                frames_dir / "court_map" / f"court_frame_{frame_num}.jpg",
            ],
        ))

    if include_detection_video:
        stages.append(_run_demo_stage(
            "detection video",
            "Render object detections over the full input clip.",
            lambda: detect_video(str(source), str(videos_dir), "all"),
            [videos_dir / "video_detection" / f"{source.stem}-detection{source.suffix}"],
        ))

    if not skip_court_video:
        trajectories_path = out_path / "trajectories" / "trajectories.json"
        stages.append(_run_demo_stage(
            "court-map video",
            "Track players, assign teams, and render top-down court positions over time.",
            lambda: map_court_video(
                str(source),
                str(videos_dir),
                train_stride=train_stride,
                debug=debug_video,
                trajectories_path=str(trajectories_path),
                max_frames=max_frames,
            ),
            [
                videos_dir / "court_map_video" / f"{source.stem}-court-map{source.suffix}",
                videos_dir / "court_map_video" / f"{source.stem}-tracking-debug{source.suffix}",
                trajectories_path,
            ],
        ))

    report_path = out_path / "report.md"
    _write_demo_report(report_path, source, out_path, frame_num, metadata, stages)
    print(f"\nDemo report written to: {report_path}")


def detect_frame(video_path: str, out_dir: str, frame_num: int = 0, filter_class: str = "all") -> None:
    """
    Detect all objects in a single frame (matches notebook cell 27).
    Outputs a summary and saves an annotated image.
    """
    import supervision as sv
    import cv2
    from . import detection
    from collections import Counter

    print(f"Loading video: {video_path}")
    out_path = Path(out_dir) / "single_frame_detection"
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        load_model_if_needed()

        with VideoLoader(video_path) as loader:
            info = loader.get_info()

            if frame_num >= info.total_frames:
                print(f"Error: Frame {frame_num} out of range (video has {info.total_frames} frames)")
                sys.exit(1)

            # Extract the specific frame
            print(f"Extracting frame {frame_num}...")
            frame = None
            for idx, f in loader:
                if idx == frame_num:
                    frame = f
                    break

            if frame is None:
                print(f"Error: Could not read frame {frame_num}")
                sys.exit(1)

            # Run detection (matching notebook exactly)
            print(f"Running detection (confidence={PLAYER_DETECTION_MODEL_CONFIDENCE}, iou={PLAYER_DETECTION_MODEL_IOU_THRESHOLD})...")
            result = detection._model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]
            detections = sv.Detections.from_inference(result)

            # Apply filter if specified
            if filter_class == "players":
                mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
                detections = detections[mask]
                print(f"Filtering to player classes only: {list(PLAYER_CLASS_IDS)}")
            elif filter_class == "numbers":
                mask = detections.class_id == NUMBER_CLASS_ID
                detections = detections[mask]
                print(f"Filtering to number class only: {NUMBER_CLASS_ID}")

            # Print summary
            print(f"\nDetections ({len(detections)} total):")
            if len(detections) > 0:
                # Count by class
                class_counts = Counter(detections.class_id)
                for class_id, count in sorted(class_counts.items()):
                    class_name = CLASS_NAMES.get(class_id, f"unknown-{class_id}")
                    suffix = "detection" if count == 1 else "detections"
                    print(f"  {class_name:24s} {count} {suffix}")
            else:
                print("  (none)")

            # Annotate frame (matching notebook style)
            color_palette = sv.ColorPalette.from_hex([
                "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
                "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
            ])
            box_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
            label_annotator = sv.LabelAnnotator(color=color_palette, text_color=sv.Color.BLACK)

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

            # Save
            out_file = out_path / f"detect_frame_{frame_num}.jpg"
            cv2.imwrite(str(out_file), annotated_frame)
            print(f"\nSaved annotated frame to {out_file}")

    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def detect_video(video_path: str, out_dir: str, filter_class: str = "all") -> None:
    """
    Run detection on full video and output annotated video.
    Uses sv.process_video() with callback (matches notebook approach).
    """
    import supervision as sv
    import subprocess
    import shutil
    from . import detection

    print(f"Running full video detection on: {video_path}")
    out_path = Path(out_dir) / "video_detection"
    out_path.mkdir(parents=True, exist_ok=True)

    source = Path(video_path)
    target = out_path / f"{source.stem}-detection{source.suffix}"

    try:
        # Load model once before processing
        load_model_if_needed()

        # Set up annotators (matching notebook style)
        color_palette = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])
        box_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
        label_annotator = sv.LabelAnnotator(color=color_palette, text_color=sv.Color.BLACK)

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            result = detection._model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
            )[0]
            detections = sv.Detections.from_inference(result)

            # Apply filter if needed
            if filter_class == "players":
                mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
                detections = detections[mask]
            elif filter_class == "numbers":
                mask = detections.class_id == NUMBER_CLASS_ID
                detections = detections[mask]

            annotated = frame.copy()
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections)
            return annotated

        print(f"Processing video (this may take a while)...")
        sv.process_video(
            source_path=str(source),
            target_path=str(target),
            callback=callback,
            show_progress=True
        )
        print(f"\nSaved detection video to {target}")

        # Compress with ffmpeg for compatibility
        if shutil.which('ffmpeg'):
            compressed_path = target.with_stem(target.stem + "_h264")
            print(f"Re-encoding with ffmpeg for compatibility...")
            result = subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', str(target),
                '-vcodec', 'libx264', '-crf', '28',
                str(compressed_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                compressed_path.replace(target)
                print(f"Saved re-encoded video to {target}")
            else:
                print(f"ffmpeg re-encode failed: {result.stderr}")
                print(f"Raw video still available at {target}")
        else:
            print("Warning: ffmpeg not found. Video may not play in all players.")

    except Exception as e:
        print(f"\nError during video detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def detect_keypoints(video_path: str, out_dir: str, frame_num: int = 0) -> None:
    """
    Detect court keypoints in a single frame (matches notebook keypoint detection cell).
    Outputs a summary and saves an annotated image with keypoint vertices.
    """
    import supervision as sv
    import cv2
    from . import homography

    print(f"Loading video: {video_path}")
    out_path = Path(out_dir) / "keypoint_detection"
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load keypoint model
        load_keypoint_model()

        with VideoLoader(video_path) as loader:
            info = loader.get_info()

            if frame_num >= info.total_frames:
                print(f"Error: Frame {frame_num} out of range (video has {info.total_frames} frames)")
                sys.exit(1)

            # Extract the specific frame
            print(f"Extracting frame {frame_num}...")
            frame = None
            for idx, f in loader:
                if idx == frame_num:
                    frame = f
                    break

            if frame is None:
                print(f"Error: Could not read frame {frame_num}")
                sys.exit(1)

            # Run keypoint detection (matching notebook exactly)
            print(f"Running keypoint detection (confidence={KEYPOINT_DETECTION_MODEL_CONFIDENCE})...")
            result = homography._keypoint_model.infer(
                frame,
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            key_points = sv.KeyPoints.from_inference(result)

            # Filter to high-confidence keypoints
            if key_points.confidence is not None and len(key_points.confidence) > 0:
                filter_mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE
                key_points = key_points[:, filter_mask]

            # Print summary
            num_keypoints = key_points.xy.shape[1] if len(key_points.xy.shape) > 1 else 0
            print(f"\nKeypoints detected: {num_keypoints} (above {KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE} confidence)")

            # Annotate frame with keypoint vertices (matching notebook style)
            keypoint_color = sv.Color.from_hex('#FF1493')
            vertex_annotator = sv.VertexAnnotator(color=keypoint_color, radius=8)

            annotated_frame = frame.copy()
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points
            )

            # Save
            out_file = out_path / f"keypoints_frame_{frame_num}.jpg"
            cv2.imwrite(str(out_file), annotated_frame)
            print(f"\nSaved annotated frame to {out_file}")

    except Exception as e:
        print(f"\nError during keypoint detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def map_court(
    video_path: str,
    out_dir: str,
    frame_num: int = 0,
    team1_color: str = "#00FF00",
    team2_color: str = "#FF0000",
    train_stride: int = 30
) -> None:
    """
    Map player positions to court coordinates for a single frame.
    Combines detection, team classification, and homography to produce:
    1. An annotated frame with team-colored bounding boxes
    2. A top-down court view showing player positions by team
    
    Args:
        train_stride: Sample every Nth frame for team classifier training (default: 30)
    
    Matches the notebook's court mapping cell.
    """
    import supervision as sv
    import cv2
    from . import detection
    from . import homography
    from sports.common.team import TeamClassifier
    from sports.common.view import ViewTransformer
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court
    from sports.common.core import MeasurementUnit

    print(f"Mapping players to court coordinates: {video_path}")
    out_path = Path(out_dir) / "court_map"
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load models
        load_model_if_needed()
        load_keypoint_model()

        # Initialize team classifier
        team_classifier = TeamClassifier(device="cpu")

        # Court configuration (matching notebook)
        config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)

        # Step 1: Collect training crops from multiple frames
        print(f"Collecting training crops (stride={train_stride})...")
        training_crops = []
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path,
            stride=train_stride
        )
        
        sample_count = 0
        for sample_frame in frame_generator:
            # Run detection on sampled frame
            result = detection._model.infer(
                sample_frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
                class_agnostic_nms=True
            )[0]
            sample_detections = sv.Detections.from_inference(result)
            
            # Filter to player classes
            player_mask = np.isin(sample_detections.class_id, list(PLAYER_CLASS_IDS))
            sample_detections = sample_detections[player_mask]
            
            if len(sample_detections) > 0:
                # Extract crops for training
                boxes = sv.scale_boxes(xyxy=sample_detections.xyxy, factor=0.4)
                for box in boxes:
                    training_crops.append(sv.crop_image(sample_frame, box))
            
            sample_count += 1
        
        print(f"  Sampled {sample_count} frames, collected {len(training_crops)} player crops")
        
        if len(training_crops) < 10:
            print(f"Warning: Only {len(training_crops)} training crops collected. Team classification may be unreliable.")
        
        # Step 2: Train the team classifier on all collected crops
        print(f"Training team classifier...")
        team_classifier.fit(training_crops)

        # Step 3: Extract the target frame
        with VideoLoader(video_path) as loader:
            info = loader.get_info()

            if frame_num >= info.total_frames:
                print(f"Error: Frame {frame_num} out of range (video has {info.total_frames} frames)")
                sys.exit(1)

            print(f"Extracting target frame {frame_num}...")
            frame = None
            for idx, f in loader:
                if idx == frame_num:
                    frame = f
                    break

            if frame is None:
                print(f"Error: Could not read frame {frame_num}")
                sys.exit(1)

            # 1. Run player detection (with class_agnostic_nms to merge duplicate detections)
            print(f"Running player detection...")
            result = detection._model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
                class_agnostic_nms=True
            )[0]
            detections = sv.Detections.from_inference(result)

            # Filter to player classes only
            player_mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
            detections = detections[player_mask]
            print(f"  Found {len(detections)} players")

            if len(detections) == 0:
                print("Error: No players detected in frame")
                sys.exit(1)

            # Assign tracker IDs (1, 2, 3, ...)
            detections.tracker_id = np.arange(1, len(detections) + 1)

            # 2. Team classification (predict using pre-trained classifier)
            print(f"Classifying teams...")
            # Scale boxes by 0.4 for better crops (matching notebook)
            boxes = sv.scale_boxes(xyxy=detections.xyxy, factor=0.4)
            crops = [sv.crop_image(frame, box) for box in boxes]

            # Predict teams using classifier trained on sampled frames
            teams = np.array(team_classifier.predict(crops))
            print(f"  Team 0: {np.sum(teams == 0)} players, Team 1: {np.sum(teams == 1)} players")

            # 3. Team-colored frame annotation
            team_colors = sv.ColorPalette.from_hex([team1_color, team2_color])
            team_box_annotator = sv.BoxAnnotator(
                color=team_colors,
                thickness=2,
                color_lookup=sv.ColorLookup.INDEX
            )

            annotated_frame = frame.copy()
            # Use tracker_id - 1 to index teams array
            annotated_frame = team_box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                custom_color_lookup=teams[detections.tracker_id - 1]
            )

            # Save team-annotated frame
            teams_file = out_path / f"teams_frame_{frame_num}.jpg"
            cv2.imwrite(str(teams_file), annotated_frame)
            print(f"\nSaved team-annotated frame to {teams_file}")

            # 4. Keypoint detection for homography
            print(f"Running keypoint detection...")
            kp_result = homography._keypoint_model.infer(
                frame,
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            key_points = sv.KeyPoints.from_inference(kp_result)

            # Filter by confidence
            if key_points.confidence is None or len(key_points.confidence) == 0:
                print("Error: No keypoints detected")
                sys.exit(1)

            landmarks_mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE
            num_landmarks = np.count_nonzero(landmarks_mask)
            print(f"  Found {num_landmarks} high-confidence keypoints")

            if num_landmarks < 4:
                print(f"Error: Need at least 4 keypoints for homography, got {num_landmarks}")
                sys.exit(1)

            # 5. Compute homography
            print(f"Computing homography...")
            court_landmarks = np.array(config.vertices)[landmarks_mask]
            frame_landmarks = key_points[:, landmarks_mask].xy[0]

            frame_to_court_transformer = ViewTransformer(
                source=frame_landmarks,
                target=court_landmarks
            )

            # 6. Transform player positions to court coordinates
            frame_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            court_xy = frame_to_court_transformer.transform_points(points=frame_xy)
            print(f"  Transformed {len(court_xy)} player positions to court coordinates")

            # 6b. Deduplicate using 5-per-team constraint
            # Basketball rule: max 5 players per team on court
            MAX_PLAYERS_PER_TEAM = 5
            
            def find_closest_pair(indices, coords):
                """Find the two closest players by court distance."""
                min_dist = float('inf')
                closest_pair = (None, None)
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices[i+1:], i+1):
                        dist = np.sqrt(np.sum((coords[idx_i] - coords[idx_j])**2))
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (idx_i, idx_j)
                return closest_pair, min_dist
            
            # Track which detections to keep
            keep_mask = np.ones(len(detections), dtype=bool)
            
            for team_id in [0, 1]:
                team_indices = np.where((teams == team_id) & keep_mask)[0]
                
                while len(team_indices) > MAX_PLAYERS_PER_TEAM:
                    # Find closest pair
                    (idx_a, idx_b), dist = find_closest_pair(team_indices, court_xy)
                    
                    # Keep higher confidence, remove lower
                    if detections.confidence[idx_a] >= detections.confidence[idx_b]:
                        remove_idx = idx_b
                    else:
                        remove_idx = idx_a
                    
                    team_label = "Team1" if team_id == 0 else "Team2"
                    print(f"  Merging duplicate {team_label} players (distance: {dist:.1f} ft) - removing player {remove_idx + 1}")
                    
                    keep_mask[remove_idx] = False
                    team_indices = np.where((teams == team_id) & keep_mask)[0]
            
            # Apply deduplication
            if not np.all(keep_mask):
                original_count = len(detections)
                detections = detections[keep_mask]
                teams = teams[keep_mask]
                court_xy = court_xy[keep_mask]
                # Re-assign tracker IDs
                detections.tracker_id = np.arange(1, len(detections) + 1)
                print(f"  After dedup: {original_count} -> {len(detections)} players ({np.sum(teams == 0)} vs {np.sum(teams == 1)})")

            # 7. Draw court with player positions
            print(f"Drawing court map...")
            court = draw_court(config=config)

            # Draw team 0 players
            team0_mask = teams == 0
            if np.any(team0_mask):
                court = draw_points_on_court(
                    config=config,
                    xy=court_xy[team0_mask],
                    fill_color=sv.Color.from_hex(team1_color),
                    court=court
                )

            # Draw team 1 players
            team1_mask = teams == 1
            if np.any(team1_mask):
                court = draw_points_on_court(
                    config=config,
                    xy=court_xy[team1_mask],
                    fill_color=sv.Color.from_hex(team2_color),
                    court=court
                )

            # Save court map
            court_file = out_path / f"court_frame_{frame_num}.jpg"
            cv2.imwrite(str(court_file), court)
            print(f"Saved court map to {court_file}")

            # Print court coordinates
            print(f"\nPlayer court coordinates (feet):")
            for i, (xy, team) in enumerate(zip(court_xy, teams)):
                team_label = "Team1" if team == 0 else "Team2"
                print(f"  Player {i+1} ({team_label}): ({xy[0]:.1f}, {xy[1]:.1f})")

    except Exception as e:
        print(f"\nError during court mapping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def map_court_video(
    video_path: str,
    out_dir: str,
    team1_color: str = "#00FF00",
    team2_color: str = "#FF0000",
    train_stride: int = 30,
    debug: bool = False,
    trajectories_path: str | None = None,
    max_frames: int | None = None
) -> None:
    """
    Map player positions to court coordinates for full video.
    Outputs a video showing just the 2D court with team-colored player dots.
    
    Uses ByteTrack for consistent player tracking across frames and assigns
    teams once per tracked player (majority vote), avoiding team color flickering.
    Enforces 5-players-per-team maximum via deduplication.
    
    Args:
        video_path: Path to input video
        out_dir: Output directory
        team1_color: Hex color for team 1 (default: green)
        team2_color: Hex color for team 2 (default: red)
        train_stride: Sample every Nth frame for team classifier training
        debug: If True, output additional video with tracking annotations
        trajectories_path: Optional path for exported court-space trajectories JSON
        max_frames: Optional maximum number of frames to process from the start
    """
    import json
    import supervision as sv
    import subprocess
    import shutil
    from tqdm import tqdm
    from collections import defaultdict
    from . import detection
    from . import homography
    from sports.common.team import TeamClassifier
    from sports.common.view import ViewTransformer
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court
    from sports.common.core import MeasurementUnit

    print(f"Mapping players to court coordinates for full video: {video_path}")
    out_path = Path(out_dir) / "court_map_video"
    out_path.mkdir(parents=True, exist_ok=True)

    source = Path(video_path)
    target = out_path / f"{source.stem}-court-map{source.suffix}"
    trajectories_file = Path(trajectories_path) if trajectories_path else out_path / f"{source.stem}-trajectories.json"

    # Constants
    MAX_PLAYERS_PER_TEAM = 5

    def find_closest_pair(indices, coords):
        """Find the two closest players by court distance."""
        min_dist = float('inf')
        closest_pair = (None, None)
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices[i+1:], i+1):
                dist = np.sqrt(np.sum((coords[idx_i] - coords[idx_j])**2))
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (idx_i, idx_j)
        return closest_pair, min_dist

    def deduplicate_teams(court_xy, teams, confidences):
        """
        Enforce max 5 players per team by removing duplicates.
        When >5 players on a team, merge closest pair (keep higher confidence).
        Returns filtered (court_xy, teams, keep_mask).
        """
        if len(court_xy) == 0:
            return court_xy, teams, np.array([], dtype=bool)
        
        keep_mask = np.ones(len(court_xy), dtype=bool)
        
        for team_id in [0, 1]:
            team_indices = np.where((teams == team_id) & keep_mask)[0]
            
            while len(team_indices) > MAX_PLAYERS_PER_TEAM:
                # Find closest pair
                (idx_a, idx_b), dist = find_closest_pair(team_indices, court_xy)
                
                if idx_a is None:
                    break
                
                # Keep higher confidence, remove lower
                if confidences[idx_a] >= confidences[idx_b]:
                    remove_idx = idx_b
                else:
                    remove_idx = idx_a
                
                keep_mask[remove_idx] = False
                team_indices = np.where((teams == team_id) & keep_mask)[0]
        
        return court_xy[keep_mask], teams[keep_mask], keep_mask

    try:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")

        # Load models
        load_model_if_needed()
        load_keypoint_model()

        # Initialize team classifier and ByteTrack tracker
        team_classifier = TeamClassifier(device="cpu")
        byte_tracker = sv.ByteTrack(frame_rate=30)

        # Court configuration (matching notebook)
        config = CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)

        # =====================================================================
        # PASS 1: Detection + ByteTrack tracking + collect crops per tracker_id
        # =====================================================================
        print(f"Pass 1: Detection + tracking...")
        video_info = sv.VideoInfo.from_video_path(video_path)
        frames_to_process = min(max_frames, video_info.total_frames) if max_frames is not None else video_info.total_frames
        
        # Collect crops for each tracker_id (for team majority voting)
        tracker_crops = defaultdict(list)  # tracker_id -> list of crops
        
        # Store frame data: list of (tracker_ids, xyxys, confidences) per frame
        frame_tracking_data = []
        
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
        
        for frame_idx, frame in tqdm(enumerate(frame_generator), total=frames_to_process, desc="Detecting+Tracking"):
            if frame_idx >= frames_to_process:
                break

            # Run player detection
            result = detection._model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
                class_agnostic_nms=True
            )[0]
            detections = sv.Detections.from_inference(result)
            
            # Filter to player classes
            player_mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
            detections = detections[player_mask]
            
            if len(detections) == 0:
                frame_tracking_data.append((np.array([]), np.array([]).reshape(0, 4), np.array([])))
                continue
            
            # Run ByteTrack to get consistent tracker IDs
            tracked = byte_tracker.update_with_detections(detections)
            
            if len(tracked) == 0:
                frame_tracking_data.append((np.array([]), np.array([]).reshape(0, 4), np.array([])))
                continue
            
            # Store tracking data for this frame
            frame_tracking_data.append((
                tracked.tracker_id.copy(),
                tracked.xyxy.copy(),
                tracked.confidence.copy()
            ))
            
            # Extract crops for team classification (scale boxes by 0.4)
            boxes = sv.scale_boxes(xyxy=tracked.xyxy, factor=0.4)
            for i, (tracker_id, box) in enumerate(zip(tracked.tracker_id, boxes)):
                crop = sv.crop_image(frame, box)
                if crop.size > 0:
                    tracker_crops[tracker_id].append(crop)
        
        print(f"  Tracked {len(tracker_crops)} unique players across {len(frame_tracking_data)} frames")
        
        # =====================================================================
        # PASS 2: Train classifier + assign teams via majority vote
        # =====================================================================
        print(f"Pass 2: Team classification...")
        
        # Collect training crops (sample from each tracker to balance)
        training_crops = []
        for tracker_id, crops in tracker_crops.items():
            # Take up to 5 crops per tracker for training
            sample_crops = crops[:5] if len(crops) > 5 else crops
            training_crops.extend(sample_crops)
        
        if len(training_crops) < 10:
            print(f"Warning: Only {len(training_crops)} training crops collected. Team classification may be unreliable.")
        
        if not training_crops:
            print("Error: No player crops found for team classification training.")
            sys.exit(1)
        
        print(f"  Training on {len(training_crops)} crops...")
        team_classifier.fit(training_crops)
        
        # Predict teams for each tracker_id via majority vote
        tracker_team_map = {}  # tracker_id -> team_id
        
        for tracker_id, crops in tracker_crops.items():
            if not crops:
                continue
            # Predict on all crops for this tracker
            predictions = team_classifier.predict(crops)
            # Majority vote
            team_id = int(np.bincount(predictions).argmax())
            tracker_team_map[tracker_id] = team_id
        
        # Count players per team
        team_counts = defaultdict(int)
        for tid, team in tracker_team_map.items():
            team_counts[team] += 1
        print(f"  Team assignments: Team0={team_counts[0]} players, Team1={team_counts[1]} players")
        
        # =====================================================================
        # PASS 3: Homography + transform + dedup + collect court positions
        #         Also render debug video if requested
        # =====================================================================
        print(f"Pass 3: Homography + court mapping{' + debug video' if debug else ''}...")
        video_xy = []  # List of (court_xy, teams) for each frame
        player_trajectories = defaultdict(list)  # tracker_id -> [[x_ft, y_ft, frame_idx], ...]
        debug_frames = [] if debug else None  # Store annotated frames for debug video
        
        # Set up debug annotators
        if debug:
            team_colors = sv.ColorPalette.from_hex([team1_color, team2_color])
            debug_box_annotator = sv.BoxAnnotator(
                color=team_colors,
                thickness=2,
                color_lookup=sv.ColorLookup.INDEX
            )
            debug_label_annotator = sv.LabelAnnotator(
                color=team_colors,
                text_color=sv.Color.WHITE,
                text_scale=0.5,
                text_padding=3,
                color_lookup=sv.ColorLookup.INDEX
            )
        
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
        skipped_frames = 0
        
        for frame_idx, frame in tqdm(enumerate(frame_generator), total=frames_to_process, desc="Mapping"):
            if frame_idx >= frames_to_process:
                break

            tracker_ids, xyxys, confidences = frame_tracking_data[frame_idx]
            
            # Handle debug annotation even for empty frames
            if debug:
                if len(tracker_ids) == 0:
                    debug_frames.append(frame.copy())
                else:
                    # Look up teams for debug annotation
                    frame_teams = np.array([tracker_team_map.get(tid, 0) for tid in tracker_ids])
                    
                    # Create sv.Detections for annotation
                    debug_detections = sv.Detections(
                        xyxy=xyxys,
                        confidence=confidences,
                        tracker_id=tracker_ids.astype(int)
                    )
                    
                    # Annotate frame with team-colored boxes and tracker IDs
                    annotated = frame.copy()
                    annotated = debug_box_annotator.annotate(
                        scene=annotated,
                        detections=debug_detections,
                        custom_color_lookup=frame_teams
                    )
                    labels = [f"#{int(tid)}" for tid in tracker_ids]
                    annotated = debug_label_annotator.annotate(
                        scene=annotated,
                        detections=debug_detections,
                        labels=labels,
                        custom_color_lookup=frame_teams
                    )
                    debug_frames.append(annotated)
            
            if len(tracker_ids) == 0:
                video_xy.append((np.array([]), np.array([])))
                skipped_frames += 1
                continue
            
            # Look up teams for each tracker
            teams = np.array([tracker_team_map.get(tid, 0) for tid in tracker_ids])
            
            # Keypoint detection for homography
            kp_result = homography._keypoint_model.infer(
                frame,
                confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE
            )[0]
            key_points = sv.KeyPoints.from_inference(kp_result)
            
            # Check for valid keypoints
            if key_points.confidence is None or len(key_points.confidence) == 0:
                video_xy.append((np.array([]), np.array([])))
                skipped_frames += 1
                continue
            
            landmarks_mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE
            
            if np.count_nonzero(landmarks_mask) < 4:
                video_xy.append((np.array([]), np.array([])))
                skipped_frames += 1
                continue
            
            # Compute homography
            court_landmarks = np.array(config.vertices)[landmarks_mask]
            frame_landmarks = key_points[:, landmarks_mask].xy[0]
            
            frame_to_court_transformer = ViewTransformer(
                source=frame_landmarks,
                target=court_landmarks
            )
            
            # Transform player positions (bottom center of bbox)
            # Compute bottom center: (x1+x2)/2, y2
            bottom_centers = np.column_stack([
                (xyxys[:, 0] + xyxys[:, 2]) / 2,  # x center
                xyxys[:, 3]  # y bottom
            ])
            court_xy = frame_to_court_transformer.transform_points(points=bottom_centers)
            
            # Apply 5-per-team deduplication
            court_xy, teams, keep_mask = deduplicate_teams(court_xy, teams, confidences)
            kept_tracker_ids = tracker_ids[keep_mask] if len(keep_mask) else np.array([])

            for tracker_id, xy in zip(kept_tracker_ids, court_xy):
                player_trajectories[str(int(tracker_id))].append([
                    float(xy[0]),
                    float(xy[1]),
                    int(frame_idx),
                ])
            
            video_xy.append((court_xy, teams))
        
        print(f"\nProcessed {len(video_xy)} frames ({skipped_frames} skipped due to insufficient keypoints/players)")

        trajectories_file.parent.mkdir(parents=True, exist_ok=True)
        trajectories_payload = {
            "game_id": source.stem,
            "source_video": str(source),
            "fps": float(video_info.fps),
            "total_frames": int(frames_to_process),
            "width": int(video_info.width),
            "height": int(video_info.height),
            "players": {
                str(int(tracker_id)): {
                    "team": int(tracker_team_map.get(int(tracker_id), -1)),
                    "trajectory": points,
                }
                for tracker_id, points in sorted(
                    player_trajectories.items(),
                    key=lambda item: int(item[0]),
                )
            },
            "ball": {"trajectory": []},
            "metadata": {
                "coordinate_system": "NBA court feet via Roboflow basketball court vertices",
                "skipped_frames": int(skipped_frames),
                "source_total_frames": int(video_info.total_frames),
                "team_classifier_train_stride": int(train_stride),
                "max_players_per_team": int(MAX_PLAYERS_PER_TEAM),
            },
        }
        with trajectories_file.open("w", encoding="utf-8") as f:
            json.dump(trajectories_payload, f, indent=2)
        print(f"Saved trajectories to {trajectories_file}")
        
        # =====================================================================
        # PASS 4: Render court-only video (and debug video if requested)
        # =====================================================================
        print(f"Rendering court map video...")
        court = draw_court(config=config)
        court_h, court_w = court.shape[:2]
        
        court_video_info = sv.VideoInfo(
            width=court_w,
            height=court_h,
            fps=video_info.fps
        )
        
        with sv.VideoSink(str(target), court_video_info) as sink:
            for court_xy, teams in tqdm(video_xy, desc="Rendering"):
                court_frame = draw_court(config=config)
                
                if len(court_xy) > 0:
                    # Draw team 0 players
                    team0_mask = teams == 0
                    if np.any(team0_mask):
                        court_frame = draw_points_on_court(
                            config=config,
                            xy=court_xy[team0_mask],
                            fill_color=sv.Color.from_hex(team1_color),
                            court=court_frame
                        )
                    
                    # Draw team 1 players
                    team1_mask = teams == 1
                    if np.any(team1_mask):
                        court_frame = draw_points_on_court(
                            config=config,
                            xy=court_xy[team1_mask],
                            fill_color=sv.Color.from_hex(team2_color),
                            court=court_frame
                        )
                
                sink.write_frame(court_frame)
        
        # Render debug video if requested
        if debug and debug_frames:
            debug_target = out_path / f"{source.stem}-tracking-debug{source.suffix}"
            print(f"Rendering debug tracking video...")
            
            debug_video_info = sv.VideoInfo(
                width=debug_frames[0].shape[1],
                height=debug_frames[0].shape[0],
                fps=video_info.fps
            )
            
            with sv.VideoSink(str(debug_target), debug_video_info) as sink:
                for debug_frame in tqdm(debug_frames, desc="Debug video"):
                    sink.write_frame(debug_frame)
            
            print(f"Saved debug video to {debug_target}")
        
        print(f"\nSaved court map video to {target}")
        
        # Re-encode with ffmpeg for compatibility
        if shutil.which('ffmpeg'):
            compressed_path = target.with_stem(target.stem + "_h264")
            print(f"Re-encoding with ffmpeg for compatibility...")
            result = subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', str(target),
                '-vcodec', 'libx264', '-crf', '28',
                str(compressed_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                compressed_path.replace(target)
                print(f"Saved re-encoded video to {target}")
            else:
                print(f"ffmpeg re-encode failed: {result.stderr}")
                print(f"Raw video still available at {target}")
        else:
            print("Warning: ffmpeg not found. Video may not play in all players.")

    except Exception as e:
        print(f"\nError during court video mapping: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def track_video(
    video_path: str,
    out_dir: str,
    filter_class: str = "players",
    sam2_checkpoint: str = None,
    sam2_config: str = None
) -> None:
    """
    Run SAM2 mask tracking on full video.
    Uses RF-DETR to detect players on frame 0, then SAM2 propagates masks.
    """
    import supervision as sv
    import subprocess
    import shutil
    from . import detection

    print(f"Running SAM2 mask tracking on: {video_path}")
    out_path = Path(out_dir) / "video_tracking"
    out_path.mkdir(parents=True, exist_ok=True)

    source = Path(video_path)
    target = out_path / f"{source.stem}-mask{source.suffix}"

    try:
        # Load RF-DETR model for initial detection
        load_model_if_needed()

        # Load SAM2 predictor
        predictor = load_sam2_predictor(sam2_checkpoint, sam2_config)
        tracker = SAM2Tracker(predictor)

        # Set up annotators
        color_palette = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
            "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
        ])
        mask_annotator = sv.MaskAnnotator(
            color=color_palette,
            color_lookup=sv.ColorLookup.TRACK,
            opacity=0.5
        )
        box_annotator = sv.BoxAnnotator(
            color=color_palette,
            color_lookup=sv.ColorLookup.TRACK,
            thickness=2
        )

        # Get first frame and run detection for SAM2 prompts
        print("Detecting objects in first frame for SAM2 prompts...")
        frame_generator = sv.get_video_frames_generator(str(source))
        first_frame = next(frame_generator)

        result = detection._model.infer(
            first_frame,
            confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
            iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
        )[0]
        detections = sv.Detections.from_inference(result)

        # Filter detections
        if filter_class == "players":
            mask = np.isin(detections.class_id, list(PLAYER_CLASS_IDS))
            detections = detections[mask]
            print(f"Filtered to {len(detections)} player detections")
        elif filter_class == "numbers":
            mask = detections.class_id == NUMBER_CLASS_ID
            detections = detections[mask]
            print(f"Filtered to {len(detections)} number detections")
        else:
            print(f"Using all {len(detections)} detections")

        if len(detections) == 0:
            print("Error: No detections found in first frame. Cannot initialize tracking.")
            sys.exit(1)

        # Assign tracker IDs
        detections.tracker_id = np.arange(1, len(detections) + 1)

        # Prompt SAM2 with first frame detections
        tracker.prompt_first_frame(first_frame, detections)

        # Process video with callback
        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            tracked = tracker.propagate(frame)
            annotated = frame.copy()
            annotated = mask_annotator.annotate(scene=annotated, detections=tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            return annotated

        print(f"Processing video with SAM2 tracking...")
        sv.process_video(
            source_path=str(source),
            target_path=str(target),
            callback=callback,
            show_progress=True
        )
        print(f"\nSaved tracking video to {target}")

        # Compress with ffmpeg for compatibility
        if shutil.which('ffmpeg'):
            compressed_path = target.with_stem(target.stem + "_h264")
            print(f"Re-encoding with ffmpeg for compatibility...")
            result = subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', str(target),
                '-vcodec', 'libx264', '-crf', '28',
                str(compressed_path)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                compressed_path.replace(target)
                print(f"Saved re-encoded video to {target}")
            else:
                print(f"ffmpeg re-encode failed: {result.stderr}")
                print(f"Raw video still available at {target}")
        else:
            print("Warning: ffmpeg not found. Video may not play in all players.")

    except Exception as e:
        print(f"\nError during SAM2 tracking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_tracking(video_path: str, out_dir: str, frames: int = 30) -> None:
    """
    Run tracking on a short video segment and render a video.
    INCLUDES TEAM ASSIGNMENT + COURT MAP.
    """
    print(
        f"Testing tracking + teams + map on {video_path} (limit {frames} frames)")
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


def test_numbers(video_path: str, out_dir: str, frames: int = 5) -> None:
    """
    Test jersey number OCR on sampled frames.
    Detects number regions and runs OCR via Roboflow hosted API.
    """
    import supervision as sv
    from . import detection

    print(
        f"Testing jersey number OCR on {video_path} (sampling {frames} frames)")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load models
        load_model_if_needed()  # Detection model
        recognizer = NumberRecognizer()  # OCR model

        with VideoLoader(video_path) as loader:
            info = loader.get_info()

            # Sample frames evenly across video
            total = info.total_frames
            stride = max(1, total // frames)

            sampled = []
            count = 0
            for idx, frame in loader:
                if idx % stride == 0 and count < frames:
                    sampled.append((idx, frame))
                    count += 1

            if not sampled:
                print("No frames sampled.")
                return

            print(f"Sampled {len(sampled)} frames")

            # Process each frame
            for frame_idx, frame in sampled:
                print(f"\n--- Frame {frame_idx} ---")
                frame_h, frame_w = frame.shape[:2]

                # Run detection
                result = detection._model.infer(
                    frame,
                    confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                    iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD
                )[0]
                detections = sv.Detections.from_inference(result)

                # Filter to number class
                number_dets, crops = extract_number_detections(
                    detections, frame, pad_px=10)

                if len(crops) == 0:
                    print(f"  No jersey numbers detected")
                    continue

                print(f"  Found {len(crops)} number regions")

                # Run OCR
                numbers = recognizer.recognize_crops(crops)

                # Print results
                for i, (num, xyxy) in enumerate(zip(numbers, number_dets.xyxy)):
                    print(
                        f"    [{i}] Number: '{num}' at bbox {tuple(map(int, xyxy))}")

                # Annotate frame
                box_annotator = sv.BoxAnnotator(thickness=2)
                label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

                annotated = frame.copy()
                annotated = box_annotator.annotate(
                    scene=annotated, detections=number_dets)
                annotated = label_annotator.annotate(
                    scene=annotated,
                    detections=number_dets,
                    labels=[f"#{n}" if n else "?" for n in numbers]
                )

                # Save
                out_file = out_path / f"numbers_frame_{frame_idx}.jpg"
                import cv2
                cv2.imwrite(str(out_file), annotated)
                print(f"  Saved: {out_file}")

    except Exception as e:
        print(f"\nError during number OCR test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def segment_possessions_cmd(
    trajectories_path: str,
    out_dir: str,
    segments_csv: str = None,
    auto_detect: bool = False,
    fastbreak_threshold: float = 4.0
) -> None:
    """
    Segment trajectory data into possessions.
    
    Args:
        trajectories_path: Path to trajectories JSON from Colab
        segments_csv: Path to manual segments CSV (optional)
        auto_detect: Auto-detect possessions from ball movement
        fastbreak_threshold: Seconds threshold for fastbreak classification
    """
    print(f"Segmenting possessions from: {trajectories_path}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load trajectories
        trajectories = load_trajectories(Path(trajectories_path))
        print(f"  Game: {trajectories.game_id}")
        print(f"  Players: {len(trajectories.players)}")
        print(f"  Frames: {trajectories.total_frames}")

        # Segment possessions
        manual_path = Path(segments_csv) if segments_csv else None
        possessions = segment_possessions(
            trajectories,
            manual_segments=manual_path,
            auto_detect=auto_detect,
            fastbreak_threshold_seconds=fastbreak_threshold
        )

        print(f"\nSegmented {len(possessions)} possessions:")
        halfcourt = sum(1 for p in possessions if p.possession_type == 'halfcourt')
        fastbreak = sum(1 for p in possessions if p.possession_type == 'fastbreak')
        print(f"  Half-court: {halfcourt}")
        print(f"  Fast breaks: {fastbreak}")

        # Save possessions
        output_file = out_path / f"{trajectories.game_id}_possessions.json"
        save_possessions(possessions, output_file)
        print(f"\nSaved possessions to: {output_file}")

    except Exception as e:
        print(f"\nError during possession segmentation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cluster_plays_cmd(
    possessions_path: str,
    out_dir: str,
    n_clusters: int = None,
    distance_threshold: float = 50.0,
    filter_type: str = None,
    use_dtw: bool = True,
    include_defense: bool = False
) -> None:
    """
    Cluster possessions into plays.
    
    Args:
        possessions_path: Path to possessions JSON
        out_dir: Output directory
        n_clusters: Fixed number of clusters (optional)
        distance_threshold: Distance threshold for clustering
        filter_type: Only cluster 'halfcourt' or 'fastbreak'
        use_dtw: Use DTW distance (vs Euclidean)
        include_defense: Include defensive trajectories in comparison
    """
    print(f"Clustering plays from: {possessions_path}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load possessions
        possessions = load_possessions(Path(possessions_path))
        print(f"  Loaded {len(possessions)} possessions")

        # Normalize
        print("Normalizing trajectories...")
        normalized = normalize_all_possessions(
            possessions,
            num_timesteps=100,
            filter_type=filter_type
        )
        print(f"  Normalized {len(normalized)} possessions")

        if len(normalized) < 2:
            print("Error: Need at least 2 possessions to cluster")
            sys.exit(1)

        # Cluster
        print("Computing distance matrix...")
        distance_matrix = compute_distance_matrix(
            normalized,
            use_dtw=use_dtw,
            include_offense=True,
            include_defense=include_defense,
            include_ball=True,
            verbose=True
        )

        print("Clustering possessions...")
        clusters = cluster_possessions(
            normalized,
            distance_matrix=distance_matrix,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            verbose=True
        )

        # Print summary
        print("\n" + get_cluster_summary(clusters))

        # Save cluster visualizations
        print("\nGenerating visualizations...")
        viz_dir = out_path / "play_visualizations"
        save_cluster_visualizations(clusters, normalized, viz_dir, top_n=10)

        # Save cluster data
        import json
        cluster_data = []
        for c in clusters:
            cluster_data.append({
                'cluster_id': c.cluster_id,
                'size': c.size,
                'possession_ids': c.possession_ids,
                'avg_intra_cluster_distance': c.avg_intra_cluster_distance
            })
        
        cluster_file = out_path / "clusters.json"
        with open(cluster_file, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        print(f"\nSaved cluster data to: {cluster_file}")

    except Exception as e:
        print(f"\nError during clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def visualize_play_cmd(
    possessions_path: str,
    out_dir: str,
    possession_id: int = None,
    cluster_id: int = None,
    clusters_path: str = None,
    render_video: bool = False,
    fps: float = 30.0
) -> None:
    """
    Visualize a specific possession or cluster.
    
    Args:
        possessions_path: Path to possessions JSON
        out_dir: Output directory
        possession_id: Specific possession to visualize
        cluster_id: Cluster to visualize (requires clusters_path)
        clusters_path: Path to clusters JSON
        render_video: Render animated video
        fps: Video FPS for rendering
    """
    import json
    import cv2
    
    print(f"Visualizing play from: {possessions_path}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load possessions
        possessions = load_possessions(Path(possessions_path))
        poss_lookup = {p.possession_id: p for p in possessions}

        if possession_id is not None:
            # Visualize single possession
            if possession_id not in poss_lookup:
                print(f"Error: Possession {possession_id} not found")
                sys.exit(1)

            poss = poss_lookup[possession_id]
            print(f"  Possession {possession_id}: frames {poss.start_frame}-{poss.end_frame}")
            print(f"  Type: {poss.possession_type}, Duration: {poss.duration_seconds:.1f}s")

            # Draw paths
            img = draw_possession_paths(poss)
            out_file = out_path / f"possession_{possession_id}.png"
            cv2.imwrite(str(out_file), img)
            print(f"  Saved: {out_file}")

            # Render video if requested
            if render_video:
                video_file = out_path / f"possession_{possession_id}.mp4"
                render_possession_video(poss, video_file, fps=fps)

            # Print video clip info
            clip_info = get_video_clip_info(poss, Path("source_video.mp4"), fps)
            print(f"\n  To extract source clip:")
            print(f"    {clip_info['ffmpeg_command']}")

        elif cluster_id is not None and clusters_path:
            # Visualize cluster
            with open(clusters_path, 'r') as f:
                clusters_data = json.load(f)

            cluster_info = None
            for c in clusters_data:
                if c['cluster_id'] == cluster_id:
                    cluster_info = c
                    break

            if cluster_info is None:
                print(f"Error: Cluster {cluster_id} not found")
                sys.exit(1)

            print(f"  Cluster {cluster_id}: {cluster_info['size']} possessions")
            print(f"  Possession IDs: {cluster_info['possession_ids']}")

            # Normalize for visualization
            normalized = normalize_all_possessions(possessions)
            norm_lookup = {p.possession_id: p for p in normalized}

            # Create PlayCluster object for visualization
            from .models import PlayCluster
            cluster = PlayCluster(
                cluster_id=cluster_id,
                possession_ids=cluster_info['possession_ids'],
                size=cluster_info['size']
            )

            # Draw cluster summary
            img = draw_cluster_summary(cluster, normalized)
            out_file = out_path / f"cluster_{cluster_id}.png"
            cv2.imwrite(str(out_file), img)
            print(f"  Saved: {out_file}")

            # List all possessions in cluster
            print(f"\n  Possessions in cluster:")
            for pid in cluster_info['possession_ids']:
                if pid in poss_lookup:
                    p = poss_lookup[pid]
                    print(f"    {pid}: frames {p.start_frame}-{p.end_frame} ({p.duration_seconds:.1f}s)")

        else:
            print("Error: Must specify either --possession-id or --cluster-id with --clusters")
            sys.exit(1)

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a basketball game into shots + box score.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'demo' command
    demo_parser = subparsers.add_parser(
        "demo", help="Run the reproducible portfolio demo")
    demo_parser.add_argument("--video", default="data/raw/clip.mp4",
                             help="Path to demo video (default: data/raw/clip.mp4)")
    demo_parser.add_argument("--out", default="data/demo",
                             help="Output directory for demo artifacts (default: data/demo)")
    demo_parser.add_argument("--frame", type=int, default=0,
                             help="Representative frame number for image outputs (default: 0)")
    demo_parser.add_argument("--train-stride", type=int, default=30,
                             help="Sample every Nth frame for team classifier training (default: 30)")
    demo_parser.add_argument("--skip-court-map", action="store_true",
                             help="Skip the single-frame court projection stage")
    demo_parser.add_argument("--skip-court-video", action="store_true",
                             help="Skip the slower full-video court map stage")
    demo_parser.add_argument("--include-detection-video", action="store_true",
                             help="Also render a full annotated detection video")
    demo_parser.add_argument("--no-debug-video", action="store_true",
                             help="Do not render the tracking debug video during court-map video generation")
    demo_parser.add_argument("--max-frames", type=int,
                             help="Limit full-video demo stages to the first N frames")

    # 'process-game' command
    process_parser = subparsers.add_parser(
        "process-game", help="Run full pipeline on a video")
    process_parser.add_argument(
        "--video", required=True, help="Path to game video")
    process_parser.add_argument(
        "--out", required=True, help="Output directory")

    # 'test-video' command
    test_parser = subparsers.add_parser(
        "test-video", help="Verify video file can be read")
    test_parser.add_argument("--video", required=True,
                             help="Path to game video")

    # 'detect-frame' command
    detect_parser = subparsers.add_parser(
        "detect-frame", help="Detect all objects in a single frame")
    detect_parser.add_argument("--video", required=True,
                               help="Path to game video")
    detect_parser.add_argument("--out", required=True,
                               help="Output directory for annotated image")
    detect_parser.add_argument("--frame", type=int, default=0,
                               help="Frame number to extract (default: 0)")
    detect_parser.add_argument("--filter", choices=["all", "players", "numbers"],
                               default="all",
                               help="Filter detections: all, players, or numbers")

    # 'detect-video' command
    detect_video_parser = subparsers.add_parser(
        "detect-video", help="Run detection on full video and output annotated video")
    detect_video_parser.add_argument("--video", required=True,
                                     help="Path to game video")
    detect_video_parser.add_argument("--out", required=True,
                                     help="Output directory for video")
    detect_video_parser.add_argument("--filter", choices=["all", "players", "numbers"],
                                     default="all",
                                     help="Filter detections: all, players, or numbers")

    # 'detect-keypoints' command
    keypoints_parser = subparsers.add_parser(
        "detect-keypoints", help="Detect court keypoints in a single frame")
    keypoints_parser.add_argument("--video", required=True,
                                  help="Path to game video")
    keypoints_parser.add_argument("--out", required=True,
                                  help="Output directory for annotated image")
    keypoints_parser.add_argument("--frame", type=int, default=0,
                                  help="Frame number to extract (default: 0)")

    # 'map-court' command
    map_court_parser = subparsers.add_parser(
        "map-court", help="Map player positions to court coordinates for a single frame")
    map_court_parser.add_argument("--video", required=True,
                                  help="Path to game video")
    map_court_parser.add_argument("--out", required=True,
                                  help="Output directory for court map images")
    map_court_parser.add_argument("--frame", type=int, default=0,
                                  help="Frame number to extract (default: 0)")
    map_court_parser.add_argument("--team1-color", default="#00FF00",
                                  help="Hex color for team 1 (default: #00FF00 green)")
    map_court_parser.add_argument("--team2-color", default="#FF0000",
                                  help="Hex color for team 2 (default: #FF0000 red)")
    map_court_parser.add_argument("--train-stride", type=int, default=30,
                                  help="Sample every Nth frame for team classifier training (default: 30)")

    # 'map-court-video' command
    map_court_video_parser = subparsers.add_parser(
        "map-court-video", help="Map player positions to court coordinates for full video")
    map_court_video_parser.add_argument("--video", required=True,
                                        help="Path to game video")
    map_court_video_parser.add_argument("--out", required=True,
                                        help="Output directory for court map video")
    map_court_video_parser.add_argument("--team1-color", default="#00FF00",
                                        help="Hex color for team 1 (default: #00FF00 green)")
    map_court_video_parser.add_argument("--team2-color", default="#FF0000",
                                        help="Hex color for team 2 (default: #FF0000 red)")
    map_court_video_parser.add_argument("--train-stride", type=int, default=30,
                                        help="Sample every Nth frame for team classifier training (default: 30)")
    map_court_video_parser.add_argument("--debug", action="store_true",
                                        help="Output additional video with tracking annotations for debugging")
    map_court_video_parser.add_argument("--trajectories-out",
                                        help="Path for exported court-space trajectories JSON")
    map_court_video_parser.add_argument("--max-frames", type=int,
                                        help="Limit processing to the first N frames")

    # 'track-video' command (SAM2 mask tracking)
    track_video_parser = subparsers.add_parser(
        "track-video", help="Run SAM2 mask tracking on full video")
    track_video_parser.add_argument("--video", required=True,
                                    help="Path to game video")
    track_video_parser.add_argument("--out", required=True,
                                    help="Output directory for video")
    track_video_parser.add_argument("--filter", choices=["all", "players", "numbers"],
                                    default="players",
                                    help="Filter detections for tracking (default: players)")
    track_video_parser.add_argument("--sam2-checkpoint",
                                    help="Path to SAM2 checkpoint (or set SAM2_CHECKPOINT env var)")
    track_video_parser.add_argument("--sam2-config",
                                    help="Path to SAM2 config yaml (or set SAM2_CONFIG env var)")

    # 'test-tracking' command
    track_parser = subparsers.add_parser(
        "test-tracking", help="Run tracking on a snippet and save video")
    track_parser.add_argument("--video", required=True,
                              help="Path to game video")
    track_parser.add_argument("--out", required=True,
                              help="Output directory for video")
    track_parser.add_argument("--frames", type=int,
                              default=30, help="Number of frames to process")

    # 'test-numbers' command
    num_parser = subparsers.add_parser(
        "test-numbers", help="Test jersey number OCR on frames")
    num_parser.add_argument("--video", required=True,
                            help="Path to game video")
    num_parser.add_argument("--out", required=True,
                            help="Output directory for images")
    num_parser.add_argument("--frames", type=int, default=5,
                            help="Number of frames to sample")

    # ===== PLAY RECOGNITION COMMANDS =====

    # 'segment-possessions' command
    segment_parser = subparsers.add_parser(
        "segment-possessions", help="Segment trajectory data into possessions")
    segment_parser.add_argument("--trajectories", required=True,
                                help="Path to trajectories JSON from Colab")
    segment_parser.add_argument("--out", required=True,
                                help="Output directory for possessions JSON")
    segment_parser.add_argument("--segments-csv",
                                help="Path to manual segments CSV (optional)")
    segment_parser.add_argument("--auto-detect", action="store_true",
                                help="Auto-detect possessions from ball movement")
    segment_parser.add_argument("--fastbreak-threshold", type=float, default=4.0,
                                help="Seconds threshold for fastbreak classification (default: 4.0)")

    # 'cluster-plays' command
    cluster_parser = subparsers.add_parser(
        "cluster-plays", help="Cluster possessions into plays")
    cluster_parser.add_argument("--possessions", required=True,
                                help="Path to possessions JSON")
    cluster_parser.add_argument("--out", required=True,
                                help="Output directory for clusters and visualizations")
    cluster_parser.add_argument("--n-clusters", type=int,
                                help="Fixed number of clusters (optional)")
    cluster_parser.add_argument("--distance-threshold", type=float, default=50.0,
                                help="Distance threshold for clustering (default: 50.0)")
    cluster_parser.add_argument("--filter-type", choices=["halfcourt", "fastbreak"],
                                help="Only cluster possessions of this type")
    cluster_parser.add_argument("--no-dtw", action="store_true",
                                help="Use Euclidean distance instead of DTW")
    cluster_parser.add_argument("--include-defense", action="store_true",
                                help="Include defensive trajectories in comparison")

    # 'visualize-play' command
    viz_parser = subparsers.add_parser(
        "visualize-play", help="Visualize a possession or cluster")
    viz_parser.add_argument("--possessions", required=True,
                            help="Path to possessions JSON")
    viz_parser.add_argument("--out", required=True,
                            help="Output directory for visualizations")
    viz_parser.add_argument("--possession-id", type=int,
                            help="Specific possession ID to visualize")
    viz_parser.add_argument("--cluster-id", type=int,
                            help="Cluster ID to visualize (requires --clusters)")
    viz_parser.add_argument("--clusters",
                            help="Path to clusters JSON")
    viz_parser.add_argument("--render-video", action="store_true",
                            help="Render animated video")
    viz_parser.add_argument("--fps", type=float, default=30.0,
                            help="Video FPS for rendering (default: 30.0)")

    args = parser.parse_args()

    if args.command == "demo":
        demo(
            args.video,
            args.out,
            frame_num=getattr(args, 'frame', 0),
            train_stride=getattr(args, 'train_stride', 30),
            skip_court_map=getattr(args, 'skip_court_map', False),
            skip_court_video=getattr(args, 'skip_court_video', False),
            include_detection_video=getattr(args, 'include_detection_video', False),
            debug_video=not getattr(args, 'no_debug_video', False),
            max_frames=getattr(args, 'max_frames', None),
        )

    elif args.command == "process-game":
        gp = GameProcessor(Path(args.video), Path(args.out))
        gp.run()

    elif args.command == "test-video":
        test_video(args.video)

    elif args.command == "detect-frame":
        detect_frame(args.video, args.out, args.frame, args.filter)

    elif args.command == "detect-video":
        detect_video(args.video, args.out, args.filter)

    elif args.command == "detect-keypoints":
        detect_keypoints(args.video, args.out, args.frame)

    elif args.command == "map-court":
        map_court(
            args.video,
            args.out,
            args.frame,
            getattr(args, 'team1_color', '#00FF00'),
            getattr(args, 'team2_color', '#FF0000'),
            getattr(args, 'train_stride', 30)
        )

    elif args.command == "map-court-video":
        map_court_video(
            args.video,
            args.out,
            getattr(args, 'team1_color', '#00FF00'),
            getattr(args, 'team2_color', '#FF0000'),
            getattr(args, 'train_stride', 30),
            getattr(args, 'debug', False),
            getattr(args, 'trajectories_out', None),
            getattr(args, 'max_frames', None)
        )

    elif args.command == "track-video":
        track_video(
            args.video,
            args.out,
            args.filter,
            getattr(args, 'sam2_checkpoint', None),
            getattr(args, 'sam2_config', None)
        )

    elif args.command == "test-tracking":
        test_tracking(args.video, args.out, args.frames)

    elif args.command == "test-numbers":
        test_numbers(args.video, args.out, args.frames)

    elif args.command == "segment-possessions":
        segment_possessions_cmd(
            args.trajectories,
            args.out,
            segments_csv=getattr(args, 'segments_csv', None),
            auto_detect=getattr(args, 'auto_detect', False),
            fastbreak_threshold=getattr(args, 'fastbreak_threshold', 4.0)
        )

    elif args.command == "cluster-plays":
        cluster_plays_cmd(
            args.possessions,
            args.out,
            n_clusters=getattr(args, 'n_clusters', None),
            distance_threshold=getattr(args, 'distance_threshold', 50.0),
            filter_type=getattr(args, 'filter_type', None),
            use_dtw=not getattr(args, 'no_dtw', False),
            include_defense=getattr(args, 'include_defense', False)
        )

    elif args.command == "visualize-play":
        visualize_play_cmd(
            args.possessions,
            args.out,
            possession_id=getattr(args, 'possession_id', None),
            cluster_id=getattr(args, 'cluster_id', None),
            clusters_path=getattr(args, 'clusters', None),
            render_video=getattr(args, 'render_video', False),
            fps=getattr(args, 'fps', 30.0)
        )


if __name__ == "__main__":
    main()
