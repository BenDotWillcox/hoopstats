from dataclasses import dataclass
from typing import Optional


@dataclass
class ShotEvent:
    period: int
    game_clock_s: float
    offense_team_id: Optional[str]
    defense_team_id: Optional[str]
    shooter_global_id: Optional[str]
    shooter_number: Optional[str]
    result: str          # "make" or "miss"
    shot_type: str       # "2PT", "3PT", "FT"
    x_ft: Optional[float]
    y_ft: Optional[float]
    distance_ft: Optional[float]
    video_frame_idx: int
