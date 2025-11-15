from typing import List, Dict, Any
from .models import ShotEvent
from .detection import Detection


def detect_shot_events(
    dets: List[Detection],
    tracks: Any,
    number_map: Dict[int, str],
    team_map: Dict[int, str],
    segment_meta: dict,
) -> List[ShotEvent]:
    """
    Use 'player-jump-shot', 'ball-in-basket', and ball trajectories
    to identify shot attempts + outcomes and assign shooter.

    Return a list of ShotEvent objects (with x_ft/y_ft left as None for now).
    """
    shot_events: List[ShotEvent] = []

    # TODO: implement logic:
    #  - find frames with 'player-jump-shot'
    #  - look in a window for 'ball-in-basket'
    #  - pick shooter via track proximity / label
    #  - fill in period, game_clock_s from segment_meta (or leave dummy)

    return shot_events
