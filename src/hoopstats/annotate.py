"""
Keyboard-driven ground-truth shot annotator.

Opens the video in an OpenCV window so shots can be marked frame-accurately:

    space      play / pause
    a / d      step 1 frame back / forward (arrow keys also work)
    s / w      jump 1 second back / forward
    m          mark a MAKE at the current frame (prompts in the terminal)
    x          mark a MISS at the current frame (prompts in the terminal)
    u          undo the most recent mark
    q / ESC    save and quit

Labels are written to a CSV after every mark (crash-safe) and reloaded on
restart, so annotation is resumable. No models or API access required.

The output schema lines up with shots.csv so `hoopstats evaluate` can match
predictions to labels directly. Location is labeled as a coarse zone rather
than court coordinates: paint, midrange, corner3, arc3 (or left blank).
"""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

GROUND_TRUTH_FIELDS = [
    "video_frame_idx",
    "time_s",
    "result",
    "offense_team_id",
    "shooter_number",
    "zone",
]

VALID_RESULTS = {"make", "miss"}
# "ft" marks a free throw; evaluation excludes these from field-goal metrics.
VALID_ZONES = {"", "paint", "midrange", "corner3", "arc3", "ft"}

# cv2.waitKeyEx arrow-key codes (Windows). Letter keys work everywhere.
KEY_LEFT = 2424832
KEY_RIGHT = 2555904


@dataclass
class ShotLabel:
    video_frame_idx: int
    time_s: float
    result: str               # "make" or "miss"
    offense_team_id: str = ""  # "0", "1", or "" if unsure
    shooter_number: str = ""   # jersey number, or "" if not visible
    zone: str = ""             # paint | midrange | corner3 | arc3 | ""


def load_labels(path: Path) -> List[ShotLabel]:
    """Load existing labels so annotation sessions are resumable."""
    if not path.exists():
        return []
    labels = []
    with path.open() as f:
        for r in csv.DictReader(f):
            labels.append(ShotLabel(
                video_frame_idx=int(r["video_frame_idx"]),
                time_s=float(r["time_s"]),
                result=r["result"],
                offense_team_id=r.get("offense_team_id", "") or "",
                shooter_number=r.get("shooter_number", "") or "",
                zone=r.get("zone", "") or "",
            ))
    return labels


def save_labels(labels: List[ShotLabel], path: Path) -> None:
    """Write all labels sorted by frame. Called after every mark."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
        writer.writeheader()
        for label in sorted(labels, key=lambda l: l.video_frame_idx):
            writer.writerow(asdict(label))


def _prompt(prompt_text: str, valid: Optional[set] = None) -> str:
    """Terminal prompt with optional validation; empty input is allowed."""
    while True:
        value = input(prompt_text).strip()
        if valid is None or value in valid:
            return value
        print(f"  Invalid value. Expected one of: {sorted(v or '(blank)' for v in valid)}")


def prompt_label_details(frame_idx: int, time_s: float, result: str) -> ShotLabel:
    """Collect team / jersey / zone for a mark via terminal prompts."""
    print(f"\n--- {result.upper()} at frame {frame_idx} ({time_s:.2f}s) ---")
    team = _prompt("  Team (0/1, blank if unsure): ", valid={"", "0", "1"})
    number = _prompt("  Jersey number (blank if not visible): ")
    zone = _prompt("  Zone (paint/midrange/corner3/arc3/ft, blank to skip): ",
                   valid=VALID_ZONES)
    return ShotLabel(
        video_frame_idx=frame_idx,
        time_s=round(time_s, 2),
        result=result,
        offense_team_id=team,
        shooter_number=number,
        zone=zone,
    )


def _draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    playing: bool,
    labels: List[ShotLabel],
) -> np.ndarray:
    """Status bar, nearby marks, and a timeline with label ticks."""
    out = frame.copy()
    h, w = out.shape[:2]

    def text(s, y, color=(255, 255, 255), scale=0.6):
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    state = "PLAYING" if playing else "PAUSED"
    text(f"frame {frame_idx}/{total_frames - 1}  {frame_idx / fps:6.2f}s  "
         f"[{state}]  marks: {len(labels)}", 25)
    text("space:play/pause  a/d:step  s/w:+/-1s  m:make  x:miss  u:undo  q:quit",
         h - 12, color=(200, 200, 200), scale=0.5)

    # Marks within ~2s of the playhead
    near = [l for l in labels if abs(l.video_frame_idx - frame_idx) <= int(2 * fps)]
    for i, l in enumerate(near[:4]):
        color = (0, 220, 0) if l.result == "make" else (0, 0, 230)
        who = f" #{l.shooter_number}" if l.shooter_number else ""
        team = f" T{l.offense_team_id}" if l.offense_team_id else ""
        text(f"{l.result.upper()}{team}{who} @ {l.video_frame_idx}",
             50 + 22 * i, color=color)

    # Timeline along the bottom
    bar_y = h - 30
    cv2.line(out, (10, bar_y), (w - 10, bar_y), (180, 180, 180), 1)
    span = max(1, total_frames - 1)
    for l in labels:
        x = 10 + int((w - 20) * l.video_frame_idx / span)
        color = (0, 220, 0) if l.result == "make" else (0, 0, 230)
        cv2.line(out, (x, bar_y - 5), (x, bar_y + 5), color, 2)
    px = 10 + int((w - 20) * frame_idx / span)
    cv2.circle(out, (px, bar_y), 4, (255, 255, 255), -1)

    return out


def annotate_video(video_path: str, out_csv: str) -> None:
    """Run the interactive annotation loop."""
    source = Path(video_path)
    out_path = Path(out_csv)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video reports no frames: {source}")

    labels = load_labels(out_path)
    if labels:
        print(f"Resuming: loaded {len(labels)} existing marks from {out_path}")
    print(f"Annotating {source.name}: {total_frames} frames @ {fps:.2f} FPS")
    print("Controls: space=play/pause  a/d=step  s/w=+/-1s  m=make  x=miss  u=undo  q=quit")

    window = "hoopstats annotate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    cap_pos = 0  # next frame VideoCapture will decode
    playing = False
    frame = None

    def read_frame(target: int) -> np.ndarray:
        """Read a frame, seeking only when stepping non-sequentially."""
        nonlocal cap_pos
        target = max(0, min(target, total_frames - 1))
        if target != cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            cap_pos = target
        ok, img = cap.read()
        if not ok:
            return None
        cap_pos = target + 1
        return img

    frame = read_frame(frame_idx)

    def step_to(target: int) -> None:
        nonlocal frame, frame_idx, playing
        playing = False
        frame_idx = max(0, min(target, total_frames - 1))
        nxt = read_frame(frame_idx)
        if nxt is not None:
            frame = nxt

    try:
        while True:
            if frame is not None:
                cv2.imshow(window, _draw_overlay(
                    frame, frame_idx, total_frames, fps, playing, labels))

            delay = max(1, int(1000 / fps)) if playing else 30
            key = cv2.waitKeyEx(delay)

            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord(' '):
                playing = not playing
            elif key in (ord('d'), KEY_RIGHT):
                step_to(frame_idx + 1)
            elif key in (ord('a'), KEY_LEFT):
                step_to(frame_idx - 1)
            elif key == ord('w'):
                step_to(frame_idx + int(fps))
            elif key == ord('s'):
                step_to(frame_idx - int(fps))
            elif key in (ord('m'), ord('x')):
                playing = False
                result = "make" if key == ord('m') else "miss"
                label = prompt_label_details(frame_idx, frame_idx / fps, result)
                labels.append(label)
                save_labels(labels, out_path)
                print(f"  Saved ({len(labels)} marks total) -> {out_path}")
            elif key == ord('u'):
                if labels:
                    removed = labels.pop()
                    save_labels(labels, out_path)
                    print(f"Removed {removed.result} at frame {removed.video_frame_idx} "
                          f"({len(labels)} marks remain)")
            elif playing:
                if frame_idx >= total_frames - 1:
                    playing = False
                else:
                    frame_idx += 1
                    nxt = read_frame(frame_idx)
                    if nxt is None:
                        playing = False
                        frame_idx -= 1
                    else:
                        frame = nxt
    finally:
        cap.release()
        cv2.destroyAllWindows()

    save_labels(labels, out_path)
    makes = sum(1 for l in labels if l.result == "make")
    print(f"\nDone. {len(labels)} shots labeled ({makes} makes, "
          f"{len(labels) - makes} misses) -> {out_path}")


def _draw_segment_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    playing: bool,
    segments: List[Tuple[int, int]],
    pending_start: Optional[int],
) -> np.ndarray:
    """Status bar + timeline showing marked live-play spans and any open in-point."""
    out = frame.copy()
    h, w = out.shape[:2]

    def text(s, y, color=(255, 255, 255), scale=0.6):
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)

    state = "PLAYING" if playing else "PAUSED"
    covered = sum(e - s for s, e in segments)
    text(f"frame {frame_idx}/{total_frames - 1}  {frame_idx / fps:6.2f}s  "
         f"[{state}]  live segments: {len(segments)} ({covered} frames)", 25)

    if pending_start is not None:
        text(f"IN-POINT open at {pending_start} ({pending_start / fps:.2f}s) "
             f"- press 'o' to close", 50, color=(0, 220, 255))
    else:
        text("press 'i' to start a live segment at this frame", 50,
             color=(200, 200, 200), scale=0.5)

    text("space:play/pause  a/d:step  s/w:+/-1s  i:in  o:out  u:undo  q:save+quit",
         h - 12, color=(200, 200, 200), scale=0.5)

    # Timeline with marked spans
    bar_y = h - 30
    cv2.line(out, (10, bar_y), (w - 10, bar_y), (180, 180, 180), 1)
    span = max(1, total_frames - 1)

    def to_x(f):
        return 10 + int((w - 20) * f / span)

    for s, e in segments:
        cv2.line(out, (to_x(s), bar_y), (to_x(e), bar_y), (0, 220, 0), 4)
    if pending_start is not None:
        cv2.line(out, (to_x(pending_start), bar_y),
                 (to_x(frame_idx), bar_y), (0, 220, 255), 4)
    cv2.circle(out, (to_x(frame_idx), bar_y), 4, (255, 255, 255), -1)

    return out


def mark_segments_video(video_path: str, out_csv: str) -> None:
    """
    Interactively mark in-scope live-play frame ranges.

    Press 'i' at the first live frame of a stretch and 'o' at the last; the pair
    becomes one [start, end) segment (end is exclusive, so the 'o' frame is
    included). Everything not inside a segment is out of scope. Segments are
    saved (sorted + merged) after every change and reloaded on restart.
    """
    from .segments import load_segments, write_segments

    source = Path(video_path)
    out_path = Path(out_csv)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video reports no frames: {source}")

    # Resume from any existing file (write_segments always tags rows "live").
    segments: List[Tuple[int, int]] = (
        load_segments(out_path) if out_path.exists() else [])
    if segments:
        print(f"Resuming: loaded {len(segments)} existing segments from {out_path}")
    print(f"Marking live-play segments on {source.name}: "
          f"{total_frames} frames @ {fps:.2f} FPS")
    print("Controls: space=play/pause  a/d=step  s/w=+/-1s  i=in  o=out  u=undo  q=save+quit")

    window = "hoopstats mark-segments"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    cap_pos = 0
    playing = False
    pending_start: Optional[int] = None

    def read_frame(target: int):
        nonlocal cap_pos
        target = max(0, min(target, total_frames - 1))
        if target != cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            cap_pos = target
        ok, img = cap.read()
        if not ok:
            return None
        cap_pos = target + 1
        return img

    frame = read_frame(frame_idx)

    def step_to(target: int):
        nonlocal frame, frame_idx, playing
        playing = False
        frame_idx = max(0, min(target, total_frames - 1))
        nxt = read_frame(frame_idx)
        if nxt is not None:
            frame = nxt

    def save():
        write_segments(segments, out_path)

    try:
        while True:
            if frame is not None:
                cv2.imshow(window, _draw_segment_overlay(
                    frame, frame_idx, total_frames, fps, playing, segments, pending_start))

            delay = max(1, int(1000 / fps)) if playing else 30
            key = cv2.waitKeyEx(delay)

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                playing = not playing
            elif key in (ord('d'), KEY_RIGHT):
                step_to(frame_idx + 1)
            elif key in (ord('a'), KEY_LEFT):
                step_to(frame_idx - 1)
            elif key == ord('w'):
                step_to(frame_idx + int(fps))
            elif key == ord('s'):
                step_to(frame_idx - int(fps))
            elif key == ord('i'):
                playing = False
                pending_start = frame_idx
                print(f"  in-point at frame {frame_idx} ({frame_idx / fps:.2f}s)")
            elif key == ord('o'):
                playing = False
                if pending_start is None:
                    print("  no open in-point; press 'i' first")
                elif frame_idx < pending_start:
                    print("  out-point is before in-point; ignored")
                else:
                    segments.append((pending_start, frame_idx + 1))  # end exclusive
                    print(f"  segment [{pending_start}, {frame_idx + 1}) added "
                          f"({len(segments)} total)")
                    pending_start = None
                    save()
            elif key == ord('u'):
                if pending_start is not None:
                    pending_start = None
                    print("  cleared open in-point")
                elif segments:
                    removed = segments.pop()
                    save()
                    print(f"  removed segment {removed} ({len(segments)} remain)")
            elif playing:
                if frame_idx >= total_frames - 1:
                    playing = False
                else:
                    frame_idx += 1
                    nxt = read_frame(frame_idx)
                    if nxt is None:
                        playing = False
                        frame_idx -= 1
                    else:
                        frame = nxt
    finally:
        cap.release()
        cv2.destroyAllWindows()

    save()
    from .segments import merge_segments
    merged = merge_segments(segments)
    covered = sum(e - s for s, e in merged)
    print(f"\nDone. {len(merged)} live segments ({covered} frames, "
          f"{covered / total_frames:.1%} of video) -> {out_path}")
