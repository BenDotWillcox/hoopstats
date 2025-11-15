from dataclasses import asdict
from pathlib import Path
from typing import List
import csv

from .models import ShotEvent


def export_shots_csv(shots: List[ShotEvent], out_dir: Path) -> Path:
    out_path = out_dir / "shots.csv"
    fieldnames = [
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
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in shots:
            writer.writerow(asdict(s))
    return out_path


def export_box_score_csv(shots_csv: Path, out_dir: Path) -> Path:
    from collections import defaultdict

    rows = []
    with shots_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    agg = defaultdict(
        lambda: {"player": "", "team": "", "fgm": 0, "fga": 0, "tpm": 0, "pts": 0})

    for r in rows:
        pid = f'{r["offense_team_id"] or ""}#{r["shooter_number"] or ""}'
        entry = agg[pid]
        entry["player"] = r["shooter_global_id"] or pid
        entry["team"] = r["offense_team_id"] or ""
        entry["fga"] += 1
        if r["result"] == "make":
            entry["fgm"] += 1
            # crude first pass for 3PT: use shot_type if set
            is_three = (r.get("shot_type") == "3PT")
            if is_three:
                entry["tpm"] += 1
                entry["pts"] += 3
            else:
                entry["pts"] += 2

    out_path = out_dir / "box_score.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["player", "team", "fgm", "fga", "tpm", "pts"])
        writer.writeheader()
        for v in agg.values():
            writer.writerow(v)

    return out_path
