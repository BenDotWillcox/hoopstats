"""
SAM2-based video tracking module.

Uses SAM2 (Segment Anything Model 2) for mask-based object tracking.
RF-DETR provides initial bounding box prompts on frame 0, then SAM2
propagates masks across subsequent frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import supervision as sv

from .config import SAM2_CHECKPOINT, SAM2_CONFIG


def load_sam2_predictor(
    checkpoint: Optional[str] = None,
    config: Optional[str] = None
):
    """
    Load SAM2 camera predictor for video tracking.
    
    Args:
        checkpoint: Path to SAM2 checkpoint file. Defaults to SAM2_CHECKPOINT env var.
        config: Path to SAM2 config yaml. Defaults to SAM2_CONFIG env var.
    
    Returns:
        SAM2 camera predictor instance
    
    Raises:
        ImportError: If sam2 package is not installed
        FileNotFoundError: If checkpoint or config files don't exist
    """
    try:
        from sam2.build_sam import build_sam2_camera_predictor
    except ImportError:
        raise ImportError(
            "SAM2 is not installed. Please install it from:\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e .\n"
            "And download checkpoints from:\n"
            "  https://github.com/facebookresearch/sam2#download-checkpoints"
        )
    
    checkpoint_path = checkpoint or SAM2_CHECKPOINT
    config_path = config or SAM2_CONFIG
    
    if not checkpoint_path:
        raise ValueError(
            "SAM2 checkpoint not specified. Set SAM2_CHECKPOINT in .env or pass --sam2-checkpoint"
        )
    if not config_path:
        raise ValueError(
            "SAM2 config not specified. Set SAM2_CONFIG in .env or pass --sam2-config"
        )
    
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
    # The config is often a Hydra config *name* resolved against SAM2's package
    # search path (e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"), not a file on
    # disk — so warn rather than fail if it isn't a local file.
    if not Path(config_path).exists():
        print(f"  Note: SAM2 config '{config_path}' is not a local file; "
              f"passing through for Hydra to resolve.")

    print(f"Loading SAM2 predictor...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")

    predictor = build_sam2_camera_predictor(str(config_path), str(checkpoint_path))
    print("SAM2 predictor loaded successfully.")
    
    return predictor


class SAM2Tracker:
    """
    SAM2-based object tracker for video.
    
    Workflow:
    1. Call prompt_first_frame() with detections from frame 0
    2. Call propagate() for each subsequent frame to track objects
    3. Call reset() to start fresh with a new video
    """
    
    def __init__(self, predictor) -> None:
        """
        Initialize tracker with a SAM2 camera predictor.
        
        Args:
            predictor: SAM2 camera predictor from build_sam2_camera_predictor()
        """
        self.predictor = predictor
        self._prompted = False

    def prompt_first_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        """
        Initialize tracking with detections from the first frame.
        
        Args:
            frame: First video frame (BGR numpy array)
            detections: Supervision Detections with bounding boxes
        
        Raises:
            ValueError: If detections is empty
        """
        if len(detections) == 0:
            raise ValueError("detections must contain at least one box")

        # Assign tracker IDs if not present
        if detections.tracker_id is None:
            detections.tracker_id = np.arange(1, len(detections) + 1)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.load_first_frame(frame)
            for xyxy, obj_id in zip(detections.xyxy, detections.tracker_id):
                bbox = np.asarray([xyxy], dtype=np.float32)
                self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=int(obj_id),
                    bbox=bbox,
                )

        self._prompted = True
        print(f"SAM2 prompted with {len(detections)} objects")

    def propagate(self, frame: np.ndarray) -> sv.Detections:
        """
        Propagate tracking to a new frame.
        
        Args:
            frame: Video frame (BGR numpy array)
        
        Returns:
            Supervision Detections with masks, bboxes, and tracker_ids
        
        Raises:
            RuntimeError: If prompt_first_frame() hasn't been called
        """
        if not self._prompted:
            raise RuntimeError("Call prompt_first_frame before propagate")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tracker_ids, mask_logits = self.predictor.track(frame)

        tracker_ids = np.asarray(tracker_ids, dtype=np.int32)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        # Ensure masks has batch dimension
        if masks.ndim == 2:
            masks = masks[None, ...]

        # Filter noisy mask segments
        masks = np.array([
            sv.filter_segments_by_distance(mask, relative_distance=0.03, mode="edge")
            for mask in masks
        ])

        # Convert masks to bounding boxes
        xyxy = sv.mask_to_xyxy(masks=masks)
        detections = sv.Detections(xyxy=xyxy, mask=masks, tracker_id=tracker_ids)
        
        return detections

    def reset(self) -> None:
        """Reset tracker state for a new video."""
        self._prompted = False
