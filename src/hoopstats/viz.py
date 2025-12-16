import cv2
import numpy as np
import subprocess
import shutil
import supervision as sv
from typing import List, Union, Dict, Optional
from pathlib import Path
from sports.basketball import CourtConfiguration, League
from sports.common.view import ViewTransformer
from sports.basketball import draw_court, draw_points_on_court

from .detection import Detection
from .tracking import TrackedObject

# Import from sports library specifically for basketball if available,
# but the notebook imports `draw_court` from `sports.basketball`.
# Let's check the import path.
# The notebook says:
# from sports.basketball import (
#     CourtConfiguration,
#     League,
#     draw_court,
#     draw_points_on_court,
#     draw_paths_on_court
# )
# We should use that.

try:
    from sports.basketball import draw_court, draw_points_on_court
except ImportError:
    # Fallback if the structure is different in installed version
    # The 'sports' library structure can vary.
    # We'll stick to what worked for CourtConfiguration above which was sports.configs.basketball?
    # Actually in homography.py I used `sports.configs.basketball`.
    # Let's try to find where draw_court is.
    # The notebook installed from git branch `feat/basketball`.
    pass


def create_annotator() -> sv.BoxAnnotator:
    return sv.BoxAnnotator(thickness=2)


def create_label_annotator() -> sv.LabelAnnotator:
    return sv.LabelAnnotator(text_color=sv.Color.BLACK)


def annotate_frame(
    frame: np.ndarray,
    detections: List[Union[Detection, TrackedObject]],
    team_map: Optional[Dict[int, int]] = None
) -> np.ndarray:
    if not detections:
        return frame

    xyxy = np.array([d.xyxy for d in detections])
    class_id = np.array([d.class_id for d in detections])
    confidence = np.array([d.score for d in detections])

    tracker_id = None
    if hasattr(detections[0], 'track_id'):
        tracker_id = np.array([d.track_id for d in detections])

    labels = []
    for d in detections:
        label = f"{d.cls} {d.score:.2f}"
        if hasattr(d, 'track_id'):
            label = f"#{d.track_id} {label}"
        if team_map and hasattr(d, 'track_id'):
            team_id = team_map.get(d.track_id)
            if team_id is not None:
                label += f" T{team_id}"
        labels.append(label)

    sv_detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        tracker_id=tracker_id
    )

    box_annotator = create_annotator()
    label_annotator = create_label_annotator()

    if team_map and tracker_id is not None:
        team_ids_array = np.array([team_map.get(tid, -1)
                                  for tid in tracker_id])
        team_palette = sv.ColorPalette.from_hex(
            ["#FF0000", "#0000FF", "#808080"])
        color_indices = np.where(team_ids_array == -1, 2, team_ids_array)

        team_box_annotator = sv.BoxAnnotator(
            color=team_palette, color_lookup=sv.ColorLookup.INDEX)
        team_label_annotator = sv.LabelAnnotator(
            color=team_palette, color_lookup=sv.ColorLookup.INDEX, text_color=sv.Color.WHITE)

        annotated_frame = frame.copy()
        annotated_frame = team_box_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            custom_color_lookup=color_indices
        )
        annotated_frame = team_label_annotator.annotate(
            scene=annotated_frame,
            detections=sv_detections,
            labels=labels,
            custom_color_lookup=color_indices
        )
        return annotated_frame

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=sv_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=sv_detections,
        labels=labels
    )

    return annotated_frame


# Cache the court configuration and base court image
_COURT_CONFIG = None
_BASE_COURT = None
# Scale for court rendering (default 20 creates ~57400x30580 image)
# 0.5 gives a more manageable ~1400x750 court image
COURT_SCALE = 0.5


def _get_court_config():
    global _COURT_CONFIG
    if _COURT_CONFIG is None:
        _COURT_CONFIG = CourtConfiguration(league=League.NBA)
    return _COURT_CONFIG


def _get_base_court():
    global _BASE_COURT
    if _BASE_COURT is None:
        _BASE_COURT = draw_court(config=_get_court_config(), scale=COURT_SCALE)
    return _BASE_COURT


def render_court_view(
    detections: List[Union[Detection, TrackedObject]],
    transformer: Optional[ViewTransformer],
    team_map: Optional[Dict[int, int]] = None
) -> np.ndarray:
    """
    Draws a top-down view of the court with players plotted.
    """
    config = _get_court_config()
    court = _get_base_court().copy()  # Copy the cached base court

    if not transformer or not detections:
        return court

    # Filter for players only (assuming class_id 3,4,5,6,7)
    # We can just try to transform everything, but plotting non-players might be weird.
    # Let's transform valid feet points.

    # Define colors matching the annotate_frame palette
    # 0=Red, 1=Blue, Other=Gray
    palette = {
        0: sv.Color.from_hex("#FF0000"),
        1: sv.Color.from_hex("#0000FF"),
        -1: sv.Color.from_hex("#808080")
    }

    # Collect points grouped by team_id
    points_by_team = {0: [], 1: [], -1: []}

    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        foot_point = np.array([[(x1 + x2) / 2, y2]])

        # Transform
        transformed = transformer.transform_points(points=foot_point)
        if transformed is None or len(transformed) == 0:
            continue

        pt = transformed[0]  # (x, y) in feet

        # Determine team
        tid = -1
        if team_map and hasattr(d, 'track_id'):
            tid = team_map.get(d.track_id, -1)

        points_by_team[tid].append(pt)

    # Draw points for each team (scale must match COURT_SCALE used in _get_base_court)
    for team_id, pts in points_by_team.items():
        if pts:
            xy = np.array(pts)
            court = draw_points_on_court(
                config=config,
                xy=xy,
                fill_color=palette[team_id],
                court=court,
                scale=COURT_SCALE,
                size=8  # smaller points for smaller court image
            )

    return court


def save_debug_frame(frame: np.ndarray, path: Path, prefix: str = "debug"):
    path.parent.mkdir(parents=True, exist_ok=True)
    filename = path / f"{prefix}.jpg"
    cv2.imwrite(str(filename), frame)
    print(f"Saved debug frame to {filename}")


def _process_frame(
    frame: np.ndarray,
    frame_idx: int,
    detections_map: dict,
    team_map: Optional[Dict[int, int]],
    homography_map: Optional[Dict[int, ViewTransformer]]
) -> np.ndarray:
    """Process a single frame with annotations and optional court view."""
    dets = detections_map.get(frame_idx, [])
    annotated_frame = annotate_frame(frame, dets, team_map=team_map)

    if homography_map:
        transformer = homography_map.get(frame_idx)

        # DEBUG: Check what we're working with
        if frame_idx == 0:
            print(f"[DEBUG] homography_map has {len(homography_map)} entries")
            print(f"[DEBUG] Frame 0 transformer: {transformer}")
            print(f"[DEBUG] Frame 0 detections: {len(dets)}")

        court_view = render_court_view(dets, transformer, team_map=team_map)

        # DEBUG: Check court dimensions
        if frame_idx == 0:
            print(
                f"[DEBUG] court_view shape: {court_view.shape}, dtype: {court_view.dtype}")
            print(
                f"[DEBUG] court_view min/max: {court_view.min()}/{court_view.max()}")
            print(f"[DEBUG] annotated_frame shape: {annotated_frame.shape}")

        h_frame, w_frame = annotated_frame.shape[:2]
        h_court, w_court = court_view.shape[:2]

        # Resize court to match frame height
        scale = h_frame / h_court
        new_w = int(w_court * scale)
        court_resized = cv2.resize(court_view, (new_w, h_frame))

        # DEBUG: Final output
        if frame_idx == 0:
            print(f"[DEBUG] court_resized shape: {court_resized.shape}")
            final = np.hstack((annotated_frame, court_resized))
            print(f"[DEBUG] final hstacked shape: {final.shape}")
            return final

        return np.hstack((annotated_frame, court_resized))

    # DEBUG: No homography branch
    if frame_idx == 0:
        print(f"[DEBUG] homography_map is falsy: {homography_map}")

    return annotated_frame


def render_video(
    frames,
    detections_map: dict,
    output_path: Path,
    fps: float = 30.0,
    team_map: Optional[Dict[int, int]] = None,
    homography_map: Optional[Dict[int, ViewTransformer]] = None
):
    """
    Render annotated video using sv.VideoSink (matches roboflow notebook).
    Re-encodes with ffmpeg for broad compatibility.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list if generator (we need to iterate twice - once for dims, once for frames)
    frames_list = list(frames)
    if not frames_list:
        print("No frames to render.")
        return

    # Process first frame to determine output dimensions
    first_idx, first_frame = frames_list[0]
    first_processed = _process_frame(
        first_frame, first_idx, detections_map, team_map, homography_map
    )
    height, width = first_processed.shape[:2]

    # Create video info for sv.VideoSink
    video_info = sv.VideoInfo(width=width, height=height, fps=fps)

    total_frames = len(frames_list)
    print(f"Rendering video to {output_path} ({total_frames} frames)...")

    with sv.VideoSink(str(output_path), video_info) as sink:
        for i, (frame_idx, frame) in enumerate(frames_list):
            final_frame = _process_frame(
                frame, frame_idx, detections_map, team_map, homography_map
            )
            sink.write_frame(final_frame)
            if (i + 1) % 10 == 0 or (i + 1) == total_frames:
                print(f"  Rendered {i + 1}/{total_frames} frames", end="\r")

    print()  # newline after progress
    print(f"Saved raw video to {output_path}")

    # Re-encode with ffmpeg for compatibility (matches notebook approach)
    if shutil.which('ffmpeg'):
        compressed_path = output_path.with_stem(output_path.stem + "_h264")
        print(f"Re-encoding with ffmpeg...")
        result = subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(output_path),
            '-vcodec', 'libx264', '-crf', '28',
            str(compressed_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Replace original with compressed version
            compressed_path.replace(output_path)
            print(f"Saved re-encoded video to {output_path}")
        else:
            print(f"ffmpeg re-encode failed: {result.stderr}")
            print(f"Raw video still available at {output_path}")
    else:
        print("Warning: ffmpeg not found. Video may not play in all players.")
        print("Install ffmpeg for better compatibility.")
