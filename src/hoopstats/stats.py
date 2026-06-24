from dataclasses import asdict
from pathlib import Path
from typing import List
import csv

from .models import ShotEvent

SHOTS_CSV_FIELDS = [
    "period",
    "game_clock_s",
    "offense_team_id",
    "defense_team_id",
    "shooter_global_id",
    "shooter_number",
    "result",
    "shot_type",
    "x_ft",
    "y_ft",
    "distance_ft",
    "video_frame_idx",
]

BOX_SCORE_FIELDS = ["player", "team", "fgm", "fga", "fg_pct", "tpm", "tpa", "pts"]


def export_shots_csv(shots: List[ShotEvent], out_dir: Path) -> Path:
    out_path = out_dir / "shots.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHOTS_CSV_FIELDS)
        writer.writeheader()
        for s in shots:
            row = {k: v for k, v in asdict(s).items() if k in SHOTS_CSV_FIELDS}
            writer.writerow(row)
    return out_path


def export_box_score_csv(shots_csv: Path, out_dir: Path) -> Path:
    """
    Aggregate shots.csv into a per-player box score.

    Players are keyed by team + jersey number; shots whose shooter could not
    be identified are bucketed into a per-team "unknown" row so the team
    totals still reconcile with the shot log.
    """
    rows = []
    with shots_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    agg = {}
    for r in rows:
        team = r["offense_team_id"] or "?"
        number = r["shooter_number"] or "unknown"
        pid = f"{team}#{number}"
        entry = agg.setdefault(pid, {
            "player": pid,
            "team": team,
            "fgm": 0,
            "fga": 0,
            "fg_pct": 0.0,
            "tpm": 0,
            "tpa": 0,
            "pts": 0,
        })

        is_three = r.get("shot_type") == "3PT"
        entry["fga"] += 1
        if is_three:
            entry["tpa"] += 1
        if r["result"] == "make":
            entry["fgm"] += 1
            if is_three:
                entry["tpm"] += 1
                entry["pts"] += 3
            else:
                entry["pts"] += 2

    for entry in agg.values():
        entry["fg_pct"] = round(entry["fgm"] / entry["fga"], 3) if entry["fga"] else 0.0

    out_path = out_dir / "box_score.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOX_SCORE_FIELDS)
        writer.writeheader()
        for key in sorted(agg, key=lambda k: (agg[k]["team"], -agg[k]["pts"], k)):
            writer.writerow(agg[key])

    return out_path
