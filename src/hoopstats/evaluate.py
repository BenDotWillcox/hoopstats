"""
Evaluate predicted shots (shots.csv) against hand-labeled ground truth
(from `hoopstats annotate`).

Metrics follow the README protocol:

- Attempt detection: precision / recall / F1, matching predictions to labels
  greedily by frame distance within a tolerance window (default 2s).
- Make/miss accuracy on matched attempts.
- Team and jersey-number accuracy on matched attempts where the label has a
  value (blank labels mean "annotator couldn't tell" and are excluded).
- Shot-type (2PT/3PT) accuracy on matched attempts with a labeled zone.

All matching/metric functions are pure so they can be unit tested without
video or models.
"""

import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .segments import Segment, frame_in_segments

# Coarse annotation zones -> expected shot type
ZONE_TO_SHOT_TYPE = {
    "paint": "2PT",
    "midrange": "2PT",
    "corner3": "3PT",
    "arc3": "3PT",
}

# Zone value marking a free throw (out of scope for field-goal metrics).
FREE_THROW_ZONE = "ft"


def match_events(
    pred_frames: List[int],
    label_frames: List[int],
    tolerance_frames: int,
) -> List[Tuple[int, int]]:
    """
    Greedily match predicted to labeled events by frame distance.

    Returns (pred_index, label_index) pairs; each event matches at most once,
    closest pairs first.
    """
    candidates = []
    for pi, pf in enumerate(pred_frames):
        for li, lf in enumerate(label_frames):
            dist = abs(pf - lf)
            if dist <= tolerance_frames:
                candidates.append((dist, pi, li))
    candidates.sort()

    matched_pred, matched_label, pairs = set(), set(), []
    for _dist, pi, li in candidates:
        if pi in matched_pred or li in matched_label:
            continue
        matched_pred.add(pi)
        matched_label.add(li)
        pairs.append((pi, li))
    return pairs


def _accuracy(pairs: List[Tuple[str, str]]) -> Tuple[Optional[float], int, int]:
    """(accuracy or None if no comparable pairs, n_correct, n_compared)."""
    if not pairs:
        return None, 0, 0
    correct = sum(1 for pred, label in pairs if pred == label)
    return correct / len(pairs), correct, len(pairs)


def _team_accuracy(pairs: List[Tuple[str, str]]) -> Tuple[Optional[float], int, int]:
    """
    Permutation-invariant team accuracy.

    The team classifier's 0/1 labels are arbitrary and not aligned to the
    annotator's convention, so team identity is only defined up to a swap.
    Report the better of the two alignments. Returns None unless the labels
    contain both teams (otherwise the metric is trivially satisfiable).
    """
    if not pairs:
        return None, 0, 0
    if len({label for _, label in pairs}) < 2:
        return None, 0, len(pairs)
    agree = sum(1 for pred, label in pairs if pred == label)
    best = max(agree, len(pairs) - agree)
    return best / len(pairs), best, len(pairs)


def _zone(row: dict) -> str:
    return (row.get("zone") or "").strip().lower()


def compute_metrics(
    preds: List[dict],
    labels: List[dict],
    fps: float = 30.0,
    tolerance_s: float = 2.0,
    segments: Optional[Sequence[Segment]] = None,
) -> dict:
    """
    Compare predicted shot rows (shots.csv schema) against ground-truth rows
    (annotate.py schema). Both are lists of CSV dicts.

    If `segments` is given, predictions and labels outside the in-scope frame
    ranges are dropped first (replays, alternate angles, dead time).

    Free throws (labels with zone == "ft") are evaluated separately: they are
    pulled out of the field-goal precision/recall, since the detector has no FT
    support. Predictions that fire on a free throw are reported as a diagnostic
    (they would inflate FGA in the box score) but are not counted as false
    positives.
    """
    tolerance_frames = int(round(tolerance_s * fps))

    # 1. Scope filtering
    n_pred_total, n_label_total = len(preds), len(labels)
    if segments is not None:
        preds = [p for p in preds
                 if frame_in_segments(int(p["video_frame_idx"]), segments)]
        labels = [l for l in labels
                  if frame_in_segments(int(l["video_frame_idx"]), segments)]
    pred_dropped = n_pred_total - len(preds)
    label_dropped = n_label_total - len(labels)

    # 2. Split free throws out of the labels
    ft_labels = [l for l in labels if _zone(l) == FREE_THROW_ZONE]
    fg_labels = [l for l in labels if _zone(l) != FREE_THROW_ZONE]

    pred_frames = [int(r["video_frame_idx"]) for r in preds]

    # 2a. Consume predictions that fired on a free throw (diagnostic, not FP)
    ft_pairs = match_events(
        pred_frames, [int(l["video_frame_idx"]) for l in ft_labels], tolerance_frames)
    ft_matched_preds = {pi for pi, _ in ft_pairs}
    ft_caught = len(ft_pairs)

    # 3. Field-goal matching over the remaining predictions
    fg_pred_idx = [i for i in range(len(preds)) if i not in ft_matched_preds]
    fg_pred_frames = [pred_frames[i] for i in fg_pred_idx]
    fg_label_frames = [int(l["video_frame_idx"]) for l in fg_labels]

    pairs_local = match_events(fg_pred_frames, fg_label_frames, tolerance_frames)
    # Remap local pred indices back to the preds list
    pairs = [(fg_pred_idx[pi], li) for pi, li in pairs_local]

    n_fg_pred, n_fg_label, n_match = len(fg_pred_idx), len(fg_labels), len(pairs)
    precision = n_match / n_fg_pred if n_fg_pred else None
    recall = n_match / n_fg_label if n_fg_label else None
    f1 = None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)

    result_pairs, team_pairs, number_pairs, type_pairs = [], [], [], []
    matched_detail = []
    for pi, li in sorted(pairs, key=lambda p: int(preds[p[0]]["video_frame_idx"])):
        p, l = preds[pi], fg_labels[li]
        result_pairs.append((p["result"], l["result"]))
        if l.get("offense_team_id"):
            team_pairs.append((p.get("offense_team_id") or "", l["offense_team_id"]))
        if l.get("shooter_number"):
            number_pairs.append((p.get("shooter_number") or "", l["shooter_number"]))
        zone = _zone(l)
        if zone in ZONE_TO_SHOT_TYPE:
            type_pairs.append((p.get("shot_type") or "", ZONE_TO_SHOT_TYPE[zone]))
        matched_detail.append({
            "pred_frame": int(p["video_frame_idx"]),
            "label_frame": int(l["video_frame_idx"]),
            "pred_result": p["result"],
            "label_result": l["result"],
            "pred_team": p.get("offense_team_id") or "",
            "label_team": l.get("offense_team_id") or "",
            "pred_number": p.get("shooter_number") or "",
            "label_number": l.get("shooter_number") or "",
            "pred_type": p.get("shot_type") or "",
            "label_zone": zone,
        })

    result_acc, result_correct, result_n = _accuracy(result_pairs)
    team_acc, team_correct, team_n = _team_accuracy(team_pairs)
    number_acc, number_correct, number_n = _accuracy(number_pairs)
    type_acc, type_correct, type_n = _accuracy(type_pairs)

    matched_pred = {pi for pi, _ in pairs}
    matched_label = {li for _, li in pairs}

    return {
        "n_pred": n_fg_pred,
        "n_label": n_fg_label,
        "n_matched": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "result_accuracy": result_acc,
        "result_correct": result_correct,
        "result_n": result_n,
        "team_accuracy": team_acc,
        "team_correct": team_correct,
        "team_n": team_n,
        "number_accuracy": number_acc,
        "number_correct": number_correct,
        "number_n": number_n,
        "shot_type_accuracy": type_acc,
        "shot_type_correct": type_correct,
        "shot_type_n": type_n,
        "false_positive_frames": sorted(
            fg_pred_frames[i] for i in range(len(fg_pred_idx))
            if fg_pred_idx[i] not in matched_pred),
        "missed_label_frames": sorted(
            fg_label_frames[i] for i in range(n_fg_label) if i not in matched_label),
        "matched": matched_detail,
        "tolerance_frames": tolerance_frames,
        "fps": fps,
        # Scope + free-throw diagnostics
        "scoped": segments is not None,
        "pred_dropped_out_of_scope": pred_dropped,
        "label_dropped_out_of_scope": label_dropped,
        "ft_labeled": len(ft_labels),
        "ft_caught_by_detector": ft_caught,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def format_report(m: dict) -> str:
    """Human-readable evaluation report (also valid markdown)."""
    lines = [
        "# Shot Detection Evaluation",
        "",
        f"Matching tolerance: +/-{m['tolerance_frames']} frames "
        f"(@ {m['fps']:.0f} FPS)",
        "",
        "Scope: field goals only (free throws excluded). "
        + ("Filtered to in-scope segments." if m["scoped"]
           else "No segments file — evaluated on all frames."),
        "",
        "## Attempt detection (field goals)",
        "",
        f"- Predicted attempts: {m['n_pred']}",
        f"- Labeled attempts:   {m['n_label']}",
        f"- Matched:            {m['n_matched']}",
        f"- Precision: {_fmt_pct(m['precision'])}",
        f"- Recall:    {_fmt_pct(m['recall'])}",
        f"- F1:        {_fmt_pct(m['f1'])}",
        "",
        "## Matched-attempt accuracy",
        "",
        f"- Make/miss:  {_fmt_pct(m['result_accuracy'])} "
        f"({m['result_correct']}/{m['result_n']})",
        f"- Team:       {_fmt_pct(m['team_accuracy'])} "
        f"({m['team_correct']}/{m['team_n']} pairs, permutation-invariant)",
        f"- Jersey #:   {_fmt_pct(m['number_accuracy'])} "
        f"({m['number_correct']}/{m['number_n']} with labeled number)",
        f"- Shot type:  {_fmt_pct(m['shot_type_accuracy'])} "
        f"({m['shot_type_correct']}/{m['shot_type_n']} with labeled zone)",
        "",
        "## Free throws & scope",
        "",
        f"- Free throws labeled (excluded from FG metrics): {m['ft_labeled']}",
        f"- Predictions that fired on a free throw (would inflate FGA): "
        f"{m['ft_caught_by_detector']}",
    ]
    if m["scoped"]:
        lines += [
            f"- Predictions dropped as out-of-scope: {m['pred_dropped_out_of_scope']}",
            f"- Labels dropped as out-of-scope: {m['label_dropped_out_of_scope']}",
        ]

    if m["false_positive_frames"]:
        lines += ["", f"False-positive attempt frames: {m['false_positive_frames']}"]
    if m["missed_label_frames"]:
        lines += ["", f"Missed labeled attempts at frames: {m['missed_label_frames']}"]

    if m["matched"]:
        lines += [
            "",
            "## Matched attempts",
            "",
            "| pred frame | label frame | result (pred/label) | team (pred/label) "
            "| number (pred/label) | type (pred/zone) |",
            "| ---: | ---: | --- | --- | --- | --- |",
        ]
        for d in m["matched"]:
            ok = "OK" if d["pred_result"] == d["label_result"] else "X"
            lines.append(
                f"| {d['pred_frame']} | {d['label_frame']} "
                f"| {d['pred_result']}/{d['label_result']} ({ok}) "
                f"| {d['pred_team'] or '-'}/{d['label_team'] or '-'} "
                f"| {d['pred_number'] or '-'}/{d['label_number'] or '-'} "
                f"| {d['pred_type'] or '-'}/{d['label_zone'] or '-'} |"
            )

    lines.append("")
    return "\n".join(lines)


def evaluate_files(
    shots_csv: str,
    labels_csv: str,
    fps: float = 30.0,
    tolerance_s: float = 2.0,
    segments_csv: Optional[str] = None,
    report_out: Optional[str] = None,
) -> dict:
    """Load both CSVs, compute metrics, print the report, optionally save it."""
    with Path(shots_csv).open() as f:
        preds = list(csv.DictReader(f))
    with Path(labels_csv).open() as f:
        labels = list(csv.DictReader(f))

    segments = None
    if segments_csv:
        from .segments import load_segments
        segments = load_segments(segments_csv)
        print(f"Loaded {len(segments)} in-scope segment(s) from {segments_csv}")

    metrics = compute_metrics(
        preds, labels, fps=fps, tolerance_s=tolerance_s, segments=segments)
    report = format_report(metrics)
    print(report)

    if report_out:
        out_path = Path(report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path}")

    return metrics
