from typing import List, Dict, Tuple, Optional
import numpy as np
import supervision as sv
from inference import get_model
from sports.common.view import ViewTransformer
from sports.basketball import CourtConfiguration, League

from .config import ROBOFLOW_API_KEY
from .detection import Detection

# Model configuration
KEYPOINT_DETECTION_MODEL_ID = "basketball-court-detection-2/14"
KEYPOINT_DETECTION_MODEL_CONFIDENCE = 0.3
KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE = 0.5

_keypoint_model = None


def load_keypoint_model():
    global _keypoint_model
    if _keypoint_model is None:
        print(f"Loading Keypoint model: {KEYPOINT_DETECTION_MODEL_ID}...")
        try:
            _keypoint_model = get_model(
                model_id=KEYPOINT_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
            print("Keypoint model loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Keypoint model: {e}")


def build_homographies(video_path: str, frames_dict: Dict[int, np.ndarray]) -> Dict[int, ViewTransformer]:
    """
    Compute homography matrix for frames where sufficient keypoints are found.
    Returns a dictionary mapping frame_idx -> ViewTransformer.

    Strategy:
    - We could run this on every frame, but it's expensive.
    - Since camera movement is continuous, we might interpolate or just compute for every frame if fast enough.
    - The notebook runs it per-frame. Let's try per-frame on the sparse set we have, or demand all frames.
    """
    load_keypoint_model()

    transformers = {}
    config = CourtConfiguration(league=League.NBA)

    print("Computing homographies...")
    for frame_idx, frame in frames_dict.items():
        try:
            result = _keypoint_model.infer(
                frame, confidence=KEYPOINT_DETECTION_MODEL_CONFIDENCE)[0]

            key_points = sv.KeyPoints.from_inference(result)

            # Filter by confidence
            if key_points.confidence is None:
                continue

            filter_mask = key_points.confidence[0] > KEYPOINT_DETECTION_MODEL_ANCHOR_CONFIDENCE

            # We need at least 4 points to compute homography
            if np.count_nonzero(filter_mask) >= 4:
                # Source points (image coords)
                frame_landmarks = key_points.xy[0][filter_mask]

                # Target points (court model coords)
                # The model outputs keypoints in a specific order that matches the config vertices?
                # Roboflow sports library handles this mapping if the model is the standard one.
                # The notebook assumes the model output aligns with config.vertices.
                court_landmarks = np.array(config.vertices)[filter_mask]

                transformer = ViewTransformer(
                    source=frame_landmarks,
                    target=court_landmarks
                )
                transformers[frame_idx] = transformer

        except Exception as e:
            # print(f"Homography failed for frame {frame_idx}: {e}")
            continue

    print(f"Computed homography for {len(transformers)} frames.")
    return transformers


def pick_homography_for_frame(transformers: Dict[int, ViewTransformer], frame_idx: int) -> Optional[ViewTransformer]:
    """
    Find the best homography for a given frame.
    Currently simple lookup. Could use nearest neighbor if sparse.
    """
    return transformers.get(frame_idx)


def add_court_coordinates(transformer: ViewTransformer, detection: Detection) -> Tuple[Optional[float], Optional[float]]:
    """
    Transform detection to court coordinates (feet).
    Returns (x, y) in feet.
    """
    # Use bottom center of the bounding box as the foot position
    x1, y1, x2, y2 = detection.xyxy
    foot_x = (x1 + x2) / 2
    foot_y = y2

    points = np.array([[foot_x, foot_y]])
    transformed = transformer.transform_points(points=points)

    if transformed is None or len(transformed) == 0:
        return None, None

    tx, ty = transformed[0]
    return float(tx), float(ty)
