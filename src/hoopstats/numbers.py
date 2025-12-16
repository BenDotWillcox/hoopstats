"""
Jersey number OCR module.

Detects jersey number regions, runs OCR, and validates readings
across frames to assign numbers to tracked players.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import supervision as sv
from inference import get_model
from sports import ConsecutiveValueTracker

from .config import ROBOFLOW_API_KEY


# Class ID for "number" in the detection model
NUMBER_CLASS_ID = 2

# OCR model configuration
NUMBER_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
NUMBER_OCR_PROMPT = "Read the number."

# Global OCR model instance
_ocr_model = None


def load_ocr_model_if_needed():
    """Lazy-load the OCR model."""
    global _ocr_model
    if _ocr_model is None:
        print(f"Loading OCR model: {NUMBER_OCR_MODEL_ID}...")
        _ocr_model = get_model(model_id=NUMBER_OCR_MODEL_ID, api_key=ROBOFLOW_API_KEY)
        print("OCR model loaded.")
    return _ocr_model


@dataclass
class NumberReading:
    """A single jersey number reading from one frame."""
    frame_idx: int
    track_id: int
    number: str
    xyxy: Tuple[int, int, int, int]  # number bbox
    confidence: float


class NumberRecognizer:
    """
    Recognizes jersey numbers from detected number regions.
    
    Uses a fine-tuned SmolVLM2 model for OCR on cropped jersey regions.
    """
    
    def __init__(self):
        self.model = load_ocr_model_if_needed()
    
    def recognize_crops(self, crops: List[np.ndarray]) -> List[str]:
        """
        Run OCR on a list of cropped number regions.
        
        Args:
            crops: List of BGR image crops (already preprocessed)
            
        Returns:
            List of recognized number strings
        """
        numbers = []
        for crop in crops:
            try:
                # Resize to expected input size
                resized = sv.resize_image(crop, resolution_wh=(224, 224))
                result = self.model.predict(resized, NUMBER_OCR_PROMPT)[0]
                numbers.append(str(result))
            except Exception as e:
                print(f"Warning: OCR failed on crop: {e}")
                numbers.append("")
        return numbers


class NumberValidator:
    """
    Validates jersey numbers over time using consecutive agreement.
    
    A number is only "confirmed" for a track_id after being read
    consistently for n_consecutive frames.
    """
    
    def __init__(self, n_consecutive: int = 3):
        """
        Args:
            n_consecutive: Number of consecutive identical readings required
        """
        self.tracker = ConsecutiveValueTracker(n_consecutive=n_consecutive)
    
    def update(self, track_ids: List[int], numbers: List[str]) -> None:
        """
        Update with new readings for this frame.
        
        Args:
            track_ids: List of player track IDs
            numbers: List of corresponding number readings
        """
        self.tracker.update(tracker_ids=track_ids, values=numbers)
    
    def get_validated(self, track_ids: List[int]) -> List[str]:
        """
        Get validated numbers for given track IDs.
        
        Args:
            track_ids: List of track IDs to query
            
        Returns:
            List of validated number strings (empty string if not yet validated)
        """
        return self.tracker.get_validated(tracker_ids=track_ids)
    
    def get_all_validated(self) -> Dict[int, str]:
        """
        Get all validated track_id -> number mappings.
        
        Returns:
            Dict mapping track_id to validated jersey number
        """
        # ConsecutiveValueTracker stores validated values internally
        # We need to access its internal state
        if hasattr(self.tracker, '_validated'):
            return dict(self.tracker._validated)
        return {}


def extract_number_detections(
    detections: sv.Detections,
    frame: np.ndarray,
    pad_px: int = 10
) -> Tuple[sv.Detections, List[np.ndarray]]:
    """
    Filter detections to only number class and extract crops.
    
    Args:
        detections: sv.Detections from the player detection model
        frame: The frame image (BGR)
        pad_px: Padding to add around number bboxes
        
    Returns:
        Tuple of (number_detections, crops)
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Filter to number class only
    number_mask = detections.class_id == NUMBER_CLASS_ID
    number_dets = detections[number_mask]
    
    if len(number_dets) == 0:
        return number_dets, []
    
    # Pad and clip bboxes
    padded = sv.pad_boxes(xyxy=number_dets.xyxy, px=pad_px, py=pad_px)
    clipped = sv.clip_boxes(padded, (frame_w, frame_h))
    
    # Extract crops
    crops = [sv.crop_image(frame, xyxy) for xyxy in clipped]
    
    return number_dets, crops


def match_numbers_to_players(
    player_detections: sv.Detections,
    number_detections: sv.Detections,
    frame_shape: Tuple[int, int],
    iou_threshold: float = 0.9
) -> List[Tuple[int, int]]:
    """
    Match detected numbers to player tracks using mask IoS.
    
    A number is matched to a player if it lies mostly inside the player bbox.
    
    Args:
        player_detections: sv.Detections with tracker_id set
        number_detections: sv.Detections for number regions
        frame_shape: (height, width) of the frame
        iou_threshold: IoS threshold for matching
        
    Returns:
        List of (player_idx, number_idx) pairs
    """
    if len(player_detections) == 0 or len(number_detections) == 0:
        return []
    
    frame_h, frame_w = frame_shape
    
    # Create masks from bboxes if not already present
    if player_detections.mask is None:
        player_masks = sv.xyxy_to_mask(
            boxes=player_detections.xyxy,
            resolution_wh=(frame_w, frame_h)
        )
    else:
        player_masks = player_detections.mask
    
    number_masks = sv.xyxy_to_mask(
        boxes=number_detections.xyxy,
        resolution_wh=(frame_w, frame_h)
    )
    
    # Compute IoS (intersection over smaller area)
    iou_matrix = sv.mask_iou_batch(
        masks_true=player_masks,
        masks_detection=number_masks,
        overlap_metric=sv.OverlapMetric.IOS
    )
    
    # Find pairs above threshold, sorted by score descending
    pairs = _coords_above_threshold(iou_matrix, iou_threshold)
    
    return pairs


def _coords_above_threshold(
    matrix: np.ndarray,
    threshold: float,
    sort_desc: bool = True
) -> List[Tuple[int, int]]:
    """
    Return all (row_idx, col_idx) where value > threshold.
    
    Optionally sort by value descending.
    """
    rows, cols = np.where(matrix > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc and pairs:
        pairs.sort(key=lambda rc: matrix[rc[0], rc[1]], reverse=True)
    return pairs


def process_frame_numbers(
    frame: np.ndarray,
    frame_idx: int,
    player_detections: sv.Detections,
    all_detections: sv.Detections,
    recognizer: NumberRecognizer,
    validator: NumberValidator,
    pad_px: int = 10,
    iou_threshold: float = 0.9
) -> List[NumberReading]:
    """
    Process jersey numbers for a single frame.
    
    1. Extract number detections and crops
    2. Run OCR on crops
    3. Match numbers to players
    4. Update the validator
    
    Args:
        frame: BGR frame image
        frame_idx: Frame index
        player_detections: sv.Detections with tracker_id for players
        all_detections: Full sv.Detections including number class
        recognizer: NumberRecognizer instance
        validator: NumberValidator instance
        pad_px: Padding for number crops
        iou_threshold: IoS threshold for matching
        
    Returns:
        List of NumberReading objects for this frame
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Extract number detections and crops
    number_dets, crops = extract_number_detections(all_detections, frame, pad_px)
    
    if len(crops) == 0:
        return []
    
    # Run OCR
    numbers = recognizer.recognize_crops(crops)
    
    # Match numbers to players
    pairs = match_numbers_to_players(
        player_detections,
        number_dets,
        (frame_h, frame_w),
        iou_threshold
    )
    
    if not pairs:
        return []
    
    # Build readings and update validator
    readings = []
    matched_track_ids = []
    matched_numbers = []
    
    for player_idx, number_idx in pairs:
        track_id = player_detections.tracker_id[player_idx]
        number_str = numbers[number_idx]
        
        if number_str:  # Only record non-empty reads
            matched_track_ids.append(int(track_id))
            matched_numbers.append(number_str)
            
            readings.append(NumberReading(
                frame_idx=frame_idx,
                track_id=int(track_id),
                number=number_str,
                xyxy=tuple(map(int, number_dets.xyxy[number_idx])),
                confidence=float(number_dets.confidence[number_idx]) if number_dets.confidence is not None else 1.0
            ))
    
    # Update validator with this frame's readings
    if matched_track_ids:
        validator.update(matched_track_ids, matched_numbers)
    
    return readings

