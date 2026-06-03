"""
Possession segmentation and trajectory normalization.

This module handles:
1. Loading trajectory JSON exported from Colab
2. Segmenting trajectories into possessions (manual or auto)
3. Normalizing trajectories for clustering
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import csv
import numpy as np
from scipy.interpolate import interp1d

from .models import (
    GameTrajectories,
    Possession,
    NormalizedPossession,
    TrajectoryPoints
)


# NBA court dimensions in feet
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
HALF_COURT_Y = COURT_LENGTH / 2  # 47 feet


def load_trajectories(json_path: Path) -> GameTrajectories:
    """
    Load trajectory data from JSON file exported by Colab notebook.
    
    Args:
        json_path: Path to trajectories JSON file
        
    Returns:
        GameTrajectories object
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return GameTrajectories.from_json(data)


def load_manual_segments(csv_path: Path) -> List[Dict]:
    """
    Load manual possession segments from CSV.
    
    Expected CSV format:
        game_id,start_frame,end_frame,type,offensive_team
        game1,450,720,halfcourt,0
        game1,800,920,fastbreak,1
    
    Args:
        csv_path: Path to segments CSV file
        
    Returns:
        List of segment dicts
    """
    segments = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'game_id': row['game_id'],
                'start_frame': int(row['start_frame']),
                'end_frame': int(row['end_frame']),
                'type': row['type'],
                'offensive_team': int(row['offensive_team'])
            })
    return segments


def _extract_trajectory_window(
    trajectory: TrajectoryPoints,
    start_frame: int,
    end_frame: int
) -> TrajectoryPoints:
    """Extract trajectory points within a frame window."""
    return [
        pt for pt in trajectory
        if start_frame <= pt[2] <= end_frame
    ]


def segment_possessions(
    trajectories: GameTrajectories,
    manual_segments: Optional[Path] = None,
    auto_detect: bool = False,
    fastbreak_threshold_seconds: float = 4.0
) -> List[Possession]:
    """
    Segment trajectory data into individual possessions.
    
    Args:
        trajectories: GameTrajectories loaded from JSON
        manual_segments: Path to CSV with manual segment definitions
        auto_detect: If True, automatically detect possessions (future)
        fastbreak_threshold_seconds: Max duration to classify as fastbreak
        
    Returns:
        List of Possession objects
    """
    possessions = []
    
    if manual_segments:
        # Use manual segments
        segments = load_manual_segments(manual_segments)
        segments = [s for s in segments if s['game_id'] == trajectories.game_id]
    elif auto_detect:
        # Auto-detect based on ball crossing half court
        segments = _auto_detect_possessions(trajectories)
    else:
        raise ValueError("Must provide manual_segments path or set auto_detect=True")
    
    # Group players by team
    team0_players = []
    team1_players = []
    for tracker_id, player_data in trajectories.players.items():
        if player_data['team'] == 0:
            team0_players.append(tracker_id)
        else:
            team1_players.append(tracker_id)
    
    # Create Possession objects
    for i, seg in enumerate(segments):
        offensive_team = seg['offensive_team']
        
        # Determine offensive and defensive players
        if offensive_team == 0:
            offense_ids = team0_players
            defense_ids = team1_players
        else:
            offense_ids = team1_players
            defense_ids = team0_players
        
        # Extract trajectories for this possession window
        offense_trajs = []
        for pid in offense_ids:
            traj = trajectories.get_player_trajectory(pid)
            window_traj = _extract_trajectory_window(traj, seg['start_frame'], seg['end_frame'])
            if window_traj:
                offense_trajs.append(window_traj)
        
        defense_trajs = []
        for pid in defense_ids:
            traj = trajectories.get_player_trajectory(pid)
            window_traj = _extract_trajectory_window(traj, seg['start_frame'], seg['end_frame'])
            if window_traj:
                defense_trajs.append(window_traj)
        
        # Ball trajectory
        ball_traj = _extract_trajectory_window(
            trajectories.ball.get('trajectory', []),
            seg['start_frame'],
            seg['end_frame']
        )
        
        # Calculate duration
        duration_frames = seg['end_frame'] - seg['start_frame']
        duration_seconds = duration_frames / trajectories.fps
        
        # Auto-classify as fastbreak if short duration
        possession_type = seg.get('type', 'halfcourt')
        if duration_seconds < fastbreak_threshold_seconds and possession_type != 'fastbreak':
            possession_type = 'fastbreak'
        
        possession = Possession(
            possession_id=i,
            game_id=trajectories.game_id,
            start_frame=seg['start_frame'],
            end_frame=seg['end_frame'],
            possession_type=possession_type,
            offensive_team=offensive_team,
            offense_trajectories=offense_trajs,
            defense_trajectories=defense_trajs,
            ball_trajectory=ball_traj,
            duration_frames=duration_frames,
            duration_seconds=duration_seconds
        )
        possessions.append(possession)
    
    return possessions


def _auto_detect_possessions(trajectories: GameTrajectories) -> List[Dict]:
    """
    Automatically detect possessions based on ball crossing half court.
    
    This is a basic implementation - can be improved with:
    - Ball direction detection
    - Made basket detection
    - Turnover detection
    
    Args:
        trajectories: GameTrajectories object
        
    Returns:
        List of segment dicts
    """
    ball_traj = trajectories.ball.get('trajectory', [])
    if not ball_traj:
        return []
    
    segments = []
    current_possession_start = None
    last_half = None  # 'top' (y < 47) or 'bottom' (y >= 47)
    
    for pt in ball_traj:
        x, y, frame = pt
        
        # Determine which half the ball is in
        current_half = 'bottom' if y >= HALF_COURT_Y else 'top'
        
        if last_half is None:
            last_half = current_half
            current_possession_start = frame
            continue
        
        # Ball crossed half court
        if current_half != last_half:
            if current_possession_start is not None:
                # End previous possession
                segments.append({
                    'game_id': trajectories.game_id,
                    'start_frame': int(current_possession_start),
                    'end_frame': int(frame),
                    'type': 'halfcourt',
                    # Team attacking 'top' half (y < 47) is team 0, 'bottom' is team 1
                    'offensive_team': 0 if last_half == 'top' else 1
                })
            
            current_possession_start = frame
            last_half = current_half
    
    # Add final possession if any
    if current_possession_start is not None and ball_traj:
        segments.append({
            'game_id': trajectories.game_id,
            'start_frame': int(current_possession_start),
            'end_frame': int(ball_traj[-1][2]),
            'type': 'halfcourt',
            'offensive_team': 0 if last_half == 'top' else 1
        })
    
    return segments


def _resample_trajectory(
    trajectory: TrajectoryPoints,
    num_points: int = 100
) -> np.ndarray:
    """
    Resample a trajectory to a fixed number of points using linear interpolation.
    
    Args:
        trajectory: List of [x, y, frame] points
        num_points: Target number of points
        
    Returns:
        numpy array of shape (num_points, 2)
    """
    if len(trajectory) < 2:
        # Not enough points - return zeros or repeat single point
        if len(trajectory) == 1:
            return np.tile([trajectory[0][0], trajectory[0][1]], (num_points, 1))
        return np.zeros((num_points, 2))
    
    # Convert to numpy
    traj_array = np.array(trajectory)
    x = traj_array[:, 0]
    y = traj_array[:, 1]
    frames = traj_array[:, 2]
    
    # Normalize frames to [0, 1]
    t_original = (frames - frames[0]) / (frames[-1] - frames[0] + 1e-6)
    t_new = np.linspace(0, 1, num_points)
    
    # Interpolate
    interp_x = interp1d(t_original, x, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(t_original, y, kind='linear', fill_value='extrapolate')
    
    resampled = np.column_stack([interp_x(t_new), interp_y(t_new)])
    return resampled


def _sort_players_by_starting_x(
    trajectories: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Sort player trajectories by their starting x-position (left to right).
    
    This ensures consistent player ordering across possessions for comparison.
    """
    if not trajectories:
        return trajectories
    
    # Get starting x for each trajectory
    start_positions = [(i, traj[0, 0] if len(traj) > 0 else 0) 
                       for i, traj in enumerate(trajectories)]
    
    # Sort by x position
    sorted_indices = [i for i, _ in sorted(start_positions, key=lambda x: x[1])]
    
    return [trajectories[i] for i in sorted_indices]


def normalize_possession(
    possession: Possession,
    num_timesteps: int = 100,
    center_on_ball: bool = True,
    mirror_to_same_direction: bool = True
) -> NormalizedPossession:
    """
    Normalize a possession's trajectories for clustering comparison.
    
    Normalization steps:
    1. Resample all trajectories to fixed length
    2. Optionally center on ball's starting position
    3. Sort offensive players by starting x-position
    4. Optionally mirror so all plays attack the same basket
    
    Args:
        possession: Raw Possession object
        num_timesteps: Number of points to resample to
        center_on_ball: If True, translate so ball starts at origin
        mirror_to_same_direction: If True, flip plays attacking different baskets
        
    Returns:
        NormalizedPossession object
    """
    # Resample trajectories
    offense_resampled = [
        _resample_trajectory(traj, num_timesteps)
        for traj in possession.offense_trajectories
    ]
    defense_resampled = [
        _resample_trajectory(traj, num_timesteps)
        for traj in possession.defense_trajectories
    ]
    ball_resampled = _resample_trajectory(possession.ball_trajectory, num_timesteps)
    
    # Compute center offset (ball's starting position)
    center_offset = np.zeros(2)
    if center_on_ball and ball_resampled is not None and len(ball_resampled) > 0:
        center_offset = ball_resampled[0].copy()
        
        # Translate all trajectories
        for i in range(len(offense_resampled)):
            offense_resampled[i] = offense_resampled[i] - center_offset
        for i in range(len(defense_resampled)):
            defense_resampled[i] = defense_resampled[i] - center_offset
        ball_resampled = ball_resampled - center_offset
    
    # Mirror if attacking "wrong" basket (standardize to attacking y > 0 direction)
    was_mirrored = False
    if mirror_to_same_direction and ball_resampled is not None and len(ball_resampled) > 1:
        # Check if ball ends up at lower y than start (attacking downward)
        if ball_resampled[-1, 1] < ball_resampled[0, 1]:
            # Mirror across x-axis (flip y)
            for i in range(len(offense_resampled)):
                offense_resampled[i][:, 1] = -offense_resampled[i][:, 1]
            for i in range(len(defense_resampled)):
                defense_resampled[i][:, 1] = -defense_resampled[i][:, 1]
            ball_resampled[:, 1] = -ball_resampled[:, 1]
            was_mirrored = True
    
    # Sort players by starting x-position
    offense_sorted = _sort_players_by_starting_x(offense_resampled)
    defense_sorted = _sort_players_by_starting_x(defense_resampled)
    
    return NormalizedPossession(
        possession_id=possession.possession_id,
        game_id=possession.game_id,
        possession_type=possession.possession_type,
        original_duration_seconds=possession.duration_seconds,
        offense_trajectories=offense_sorted,
        defense_trajectories=defense_sorted,
        ball_trajectory=ball_resampled,
        center_offset=center_offset,
        was_mirrored=was_mirrored,
        num_timesteps=num_timesteps
    )


def normalize_all_possessions(
    possessions: List[Possession],
    num_timesteps: int = 100,
    filter_type: Optional[str] = None
) -> List[NormalizedPossession]:
    """
    Normalize all possessions for clustering.
    
    Args:
        possessions: List of raw Possession objects
        num_timesteps: Number of points to resample to
        filter_type: If set, only include possessions of this type ('halfcourt' or 'fastbreak')
        
    Returns:
        List of NormalizedPossession objects
    """
    normalized = []
    for poss in possessions:
        if filter_type and poss.possession_type != filter_type:
            continue
        normalized.append(normalize_possession(poss, num_timesteps))
    return normalized


def save_possessions(possessions: List[Possession], output_path: Path) -> None:
    """
    Save possessions to JSON file.
    
    Args:
        possessions: List of Possession objects
        output_path: Path to output JSON file
    """
    data = []
    for poss in possessions:
        data.append({
            'possession_id': poss.possession_id,
            'game_id': poss.game_id,
            'start_frame': poss.start_frame,
            'end_frame': poss.end_frame,
            'type': poss.possession_type,
            'offensive_team': poss.offensive_team,
            'duration_frames': poss.duration_frames,
            'duration_seconds': poss.duration_seconds,
            'offense_trajectories': poss.offense_trajectories,
            'defense_trajectories': poss.defense_trajectories,
            'ball_trajectory': poss.ball_trajectory
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_possessions(json_path: Path) -> List[Possession]:
    """
    Load possessions from JSON file.
    
    Args:
        json_path: Path to possessions JSON file
        
    Returns:
        List of Possession objects
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    possessions = []
    for item in data:
        poss = Possession(
            possession_id=item['possession_id'],
            game_id=item['game_id'],
            start_frame=item['start_frame'],
            end_frame=item['end_frame'],
            possession_type=item['type'],
            offensive_team=item['offensive_team'],
            offense_trajectories=item['offense_trajectories'],
            defense_trajectories=item['defense_trajectories'],
            ball_trajectory=item['ball_trajectory'],
            duration_frames=item['duration_frames'],
            duration_seconds=item['duration_seconds']
        )
        possessions.append(poss)
    
    return possessions
