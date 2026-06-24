"""
In-scope segment ranges for an uncut broadcast.

A segments CSV marks the frame ranges the pipeline is meant to handle (the main
tactical camera during live play) and lets us ignore replays, alternate angles,
and dead time. Frame indices are absolute in the original video, so they line
up with both `shots.csv` predictions and `annotate` ground truth.

CSV format (header required):

    start_frame,end_frame,type
    0,1830,live
    1830,2400,replay
    2400,5200,live

`type` is optional. If present, only rows whose type is in `scope_types`
(default {"live"}) are in scope; everything else is excluded. If the column is
absent, every row is treated as in scope.
"""

import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Segment = Tuple[int, int]

DEFAULT_SCOPE_TYPES = ("live",)


def load_segments(
    path: str | Path,
    scope_types: Sequence[str] = DEFAULT_SCOPE_TYPES,
) -> List[Segment]:
    """Load in-scope (start_frame, end_frame) ranges from a segments CSV."""
    path = Path(path)
    scope = {t.lower() for t in scope_types}
    segments: List[Segment] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        has_type = reader.fieldnames is not None and "type" in reader.fieldnames
        for row in reader:
            if has_type:
                row_type = (row.get("type") or "").strip().lower()
                if row_type and row_type not in scope:
                    continue
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            if end <= start:
                raise ValueError(
                    f"Segment end_frame ({end}) must be > start_frame ({start})")
            segments.append((start, end))
    return merge_segments(segments)


def merge_segments(segments: Sequence[Segment]) -> List[Segment]:
    """Sort and merge overlapping/adjacent ranges so filtering is unambiguous."""
    if not segments:
        return []
    ordered = sorted(segments)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def write_segments(segments: Sequence[Segment], path: str | Path) -> Path:
    """Write segments to a CSV (start_frame,end_frame,type=live), sorted/merged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_frame", "end_frame", "type"])
        for start, end in merge_segments(segments):
            writer.writerow([start, end, "live"])
    return path


def frame_in_segments(frame_idx: int, segments: Sequence[Segment]) -> bool:
    """True if frame_idx falls in any [start, end) range."""
    return any(start <= frame_idx < end for start, end in segments)


def segment_index(frame_idx: int, segments: Sequence[Segment]) -> Optional[int]:
    """Index of the segment containing frame_idx, or None if out of scope."""
    for i, (start, end) in enumerate(segments):
        if start <= frame_idx < end:
            return i
    return None
