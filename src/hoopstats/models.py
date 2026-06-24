from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


# Type alias for trajectory data: list of [x, y, frame] points
TrajectoryPoints = List[List[float]]


@dataclass
class GameTrajectories:
    """Raw trajectory data exported from Colab processing."""
    game_id: str
    source_video: str
    fps: float
    total_frames: int
    width: int
    height: int
    players: Dict[str, Dict]  # tracker_id -> {"team": int, "trajectory": [[x,y,frame],...]}
    ball: Dict  # {"trajectory": [[x,y,frame],...]}
    
    @classmethod
    def from_json(cls, data: dict) -> "GameTrajectories":
        """Load from JSON dict."""
        return cls(
            game_id=data["game_id"],
            source_video=data["source_video"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            players=data["players"],
            ball=data["ball"]
        )
    
    def get_player_team(self, tracker_id: str) -> int:
        """Get team ID for a player tracker."""
        return self.players.get(tracker_id, {}).get("team", -1)
    
    def get_player_trajectory(self, tracker_id: str) -> TrajectoryPoints:
        """Get trajectory for a player tracker."""
        return self.players.get(tracker_id, {}).get("trajectory", [])


@dataclass
class Possession:
    """A single possession extracted from game trajectories."""
    possession_id: int
    game_id: str
    start_frame: int
    end_frame: int
    possession_type: str  # "halfcourt" or "fastbreak"
    offensive_team: int  # 0 or 1
    
    # Trajectories for this possession window
    # Each is a list of [x, y, frame] points within the possession timeframe
    offense_trajectories: List[TrajectoryPoints] = field(default_factory=list)  # 5 players
    defense_trajectories: List[TrajectoryPoints] = field(default_factory=list)  # 5 players
    ball_trajectory: TrajectoryPoints = field(default_factory=list)
    
    # Metadata
    duration_frames: int = 0
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        self.duration_frames = self.end_frame - self.start_frame
    
    @property
    def num_offense_players(self) -> int:
        return len(self.offense_trajectories)
    
    @property
    def num_defense_players(self) -> int:
        return len(self.defense_trajectories)


@dataclass
class NormalizedPossession:
    """
    A possession with normalized trajectories for clustering.
    
    Normalization includes:
    - Time resampled to fixed length (e.g., 100 points)
    - Positions normalized (centered, mirrored to same direction)
    - Players sorted by starting position
    """
    possession_id: int
    game_id: str
    possession_type: str
    original_duration_seconds: float
    
    # Normalized trajectories as numpy arrays
    # Shape: (num_timesteps, 2) for each trajectory
    offense_trajectories: List[np.ndarray] = field(default_factory=list)  # 5 players
    defense_trajectories: List[np.ndarray] = field(default_factory=list)  # 5 players  
    ball_trajectory: Optional[np.ndarray] = None
    
    # Normalization parameters (for inverse transform if needed)
    center_offset: Optional[np.ndarray] = None
    was_mirrored: bool = False
    num_timesteps: int = 100


@dataclass
class PlayCluster:
    """A cluster of similar possessions representing a 'play'."""
    cluster_id: int
    possession_ids: List[int] = field(default_factory=list)
    
    # Cluster statistics
    size: int = 0
    
    # Representative trajectory (centroid/average)
    centroid_offense: List[np.ndarray] = field(default_factory=list)
    centroid_defense: List[np.ndarray] = field(default_factory=list)
    centroid_ball: Optional[np.ndarray] = None
    
    # Clustering metadata
    avg_intra_cluster_distance: float = 0.0
    
    def __post_init__(self):
        self.size = len(self.possession_ids)
    
    @property
    def usage_count(self) -> int:
        """Number of times this play was run."""
        return self.size


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
    # Image-space shooter box (xyxy) at the release frame; used for the
    # homography projection and excluded from CSV export.
    shooter_xyxy: Optional[tuple] = None
