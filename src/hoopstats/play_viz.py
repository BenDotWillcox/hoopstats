"""
Play visualization for basketball possession analysis.

This module provides visualization tools for:
- Single possession replay (animated)
- Cluster summary (spaghetti plots)
- Average trajectory visualization
- Video clip extraction references
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

from .models import Possession, NormalizedPossession, PlayCluster

# Try to import supervision and sports libraries
try:
    import supervision as sv
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court, draw_paths_on_court
    from sports.common.core import MeasurementUnit
    HAS_VISUALIZATION_LIBS = True
except ImportError:
    HAS_VISUALIZATION_LIBS = False


# NBA court dimensions
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0

# Default colors
TEAM_COLORS = {
    0: (0, 255, 0),    # Green (BGR)
    1: (0, 0, 255),    # Red (BGR)
}
BALL_COLOR = (0, 165, 255)  # Orange (BGR)


def _get_court_config():
    """Get NBA court configuration."""
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    return CourtConfiguration(league=League.NBA, measurement_unit=MeasurementUnit.FEET)


def draw_possession_frame(
    possession: Possession,
    frame_idx: int,
    court_image: Optional[np.ndarray] = None,
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ball_color: Tuple[int, int, int] = BALL_COLOR,
    point_radius: int = 15
) -> np.ndarray:
    """
    Draw a single frame of a possession on the court.
    
    Args:
        possession: Possession object
        frame_idx: Frame index within possession (relative to start_frame)
        court_image: Pre-drawn court image (optional, will create if None)
        team_colors: Dict mapping team_id -> BGR color tuple
        ball_color: BGR color for ball
        point_radius: Radius of player dots
        
    Returns:
        Court image with players and ball drawn
    """
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    
    if team_colors is None:
        team_colors = TEAM_COLORS
    
    config = _get_court_config()
    
    if court_image is None:
        court = draw_court(config=config)
    else:
        court = court_image.copy()
    
    # Absolute frame number
    abs_frame = possession.start_frame + frame_idx
    
    # Draw offensive players
    offense_color = sv.Color.from_rgb_tuple(team_colors.get(possession.offensive_team, (0, 255, 0)))
    offense_points = []
    for traj in possession.offense_trajectories:
        # Find position at this frame
        for pt in traj:
            if int(pt[2]) == abs_frame:
                offense_points.append([pt[0], pt[1]])
                break
    
    if offense_points:
        court = draw_points_on_court(
            config=config,
            xy=np.array(offense_points),
            fill_color=offense_color,
            court=court
        )
    
    # Draw defensive players
    defense_team = 1 - possession.offensive_team
    defense_color = sv.Color.from_rgb_tuple(team_colors.get(defense_team, (0, 0, 255)))
    defense_points = []
    for traj in possession.defense_trajectories:
        for pt in traj:
            if int(pt[2]) == abs_frame:
                defense_points.append([pt[0], pt[1]])
                break
    
    if defense_points:
        court = draw_points_on_court(
            config=config,
            xy=np.array(defense_points),
            fill_color=defense_color,
            court=court
        )
    
    # Draw ball
    ball_sv_color = sv.Color.from_rgb_tuple(ball_color)
    for pt in possession.ball_trajectory:
        if int(pt[2]) == abs_frame:
            court = draw_points_on_court(
                config=config,
                xy=np.array([[pt[0], pt[1]]]),
                fill_color=ball_sv_color,
                court=court,
                radius=point_radius // 2  # Smaller ball
            )
            break
    
    return court


def draw_possession_paths(
    possession: Possession,
    court_image: Optional[np.ndarray] = None,
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ball_color: Tuple[int, int, int] = BALL_COLOR,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw all trajectory paths for a possession (spaghetti plot).
    
    Args:
        possession: Possession object
        court_image: Pre-drawn court image (optional)
        team_colors: Dict mapping team_id -> BGR color tuple
        ball_color: BGR color for ball
        line_thickness: Thickness of trajectory lines
        
    Returns:
        Court image with paths drawn
    """
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    
    if team_colors is None:
        team_colors = TEAM_COLORS
    
    config = _get_court_config()
    
    if court_image is None:
        court = draw_court(config=config)
    else:
        court = court_image.copy()
    
    # Draw offensive paths
    offense_color = sv.Color.from_rgb_tuple(team_colors.get(possession.offensive_team, (0, 255, 0)))
    for traj in possession.offense_trajectories:
        if len(traj) > 1:
            path = np.array([[pt[0], pt[1]] for pt in traj])
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=offense_color,
                court=court,
                thickness=line_thickness
            )
    
    # Draw defensive paths
    defense_team = 1 - possession.offensive_team
    defense_color = sv.Color.from_rgb_tuple(team_colors.get(defense_team, (0, 0, 255)))
    for traj in possession.defense_trajectories:
        if len(traj) > 1:
            path = np.array([[pt[0], pt[1]] for pt in traj])
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=defense_color,
                court=court,
                thickness=line_thickness
            )
    
    # Draw ball path
    if len(possession.ball_trajectory) > 1:
        ball_sv_color = sv.Color.from_rgb_tuple(ball_color)
        ball_path = np.array([[pt[0], pt[1]] for pt in possession.ball_trajectory])
        court = draw_paths_on_court(
            config=config,
            paths=[ball_path],
            color=ball_sv_color,
            court=court,
            thickness=line_thickness
        )
    
    return court


def draw_normalized_possession_paths(
    possession: NormalizedPossession,
    court_image: Optional[np.ndarray] = None,
    offense_color: Tuple[int, int, int] = (0, 255, 0),
    defense_color: Tuple[int, int, int] = (0, 0, 255),
    ball_color: Tuple[int, int, int] = BALL_COLOR,
    line_thickness: int = 2,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Draw paths for a normalized possession.
    
    Args:
        possession: NormalizedPossession object
        court_image: Pre-drawn court image (optional)
        offense_color: BGR color for offense
        defense_color: BGR color for defense
        ball_color: BGR color for ball
        line_thickness: Line thickness
        alpha: Opacity (0-1)
        
    Returns:
        Court image with paths drawn
    """
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    
    config = _get_court_config()
    
    if court_image is None:
        court = draw_court(config=config)
    else:
        court = court_image.copy()
    
    # Note: Normalized trajectories are centered and may be mirrored
    # We need to un-center them for display on the actual court
    # For now, we'll add back a reasonable offset (center of half-court)
    center_offset = np.array([COURT_WIDTH_FT / 2, COURT_LENGTH_FT * 0.75])
    
    # Draw offensive paths
    sv_offense = sv.Color.from_rgb_tuple(offense_color)
    for traj in possession.offense_trajectories:
        if len(traj) > 1:
            path = traj + center_offset
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=sv_offense,
                court=court,
                thickness=line_thickness
            )
    
    # Draw defensive paths
    sv_defense = sv.Color.from_rgb_tuple(defense_color)
    for traj in possession.defense_trajectories:
        if len(traj) > 1:
            path = traj + center_offset
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=sv_defense,
                court=court,
                thickness=line_thickness
            )
    
    # Draw ball path
    if possession.ball_trajectory is not None and len(possession.ball_trajectory) > 1:
        sv_ball = sv.Color.from_rgb_tuple(ball_color)
        ball_path = possession.ball_trajectory + center_offset
        court = draw_paths_on_court(
            config=config,
            paths=[ball_path],
            color=sv_ball,
            court=court,
            thickness=line_thickness
        )
    
    return court


def draw_cluster_summary(
    cluster: PlayCluster,
    possessions: List[NormalizedPossession],
    court_image: Optional[np.ndarray] = None,
    offense_color: Tuple[int, int, int] = (0, 255, 0),
    defense_color: Tuple[int, int, int] = (0, 0, 255),
    centroid_color: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 1,
    centroid_thickness: int = 3,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw a cluster summary showing all possessions and the centroid.
    
    Args:
        cluster: PlayCluster object
        possessions: List of all NormalizedPossession objects
        court_image: Pre-drawn court image (optional)
        offense_color: BGR color for offense paths
        defense_color: BGR color for defense paths
        centroid_color: BGR color for centroid paths
        line_thickness: Thickness for individual paths
        centroid_thickness: Thickness for centroid path
        alpha: Opacity for individual paths
        
    Returns:
        Court image with cluster visualization
    """
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    
    config = _get_court_config()
    
    if court_image is None:
        court = draw_court(config=config)
    else:
        court = court_image.copy()
    
    # Create possession lookup
    poss_lookup = {p.possession_id: p for p in possessions}
    
    # Draw individual possession paths (faded)
    for poss_id in cluster.possession_ids:
        if poss_id in poss_lookup:
            poss = poss_lookup[poss_id]
            # Blend with alpha
            overlay = draw_normalized_possession_paths(
                poss,
                court_image=draw_court(config=config),
                offense_color=offense_color,
                defense_color=defense_color,
                line_thickness=line_thickness
            )
            court = cv2.addWeighted(court, 1.0, overlay, alpha, 0)
    
    # Draw centroid paths (bold)
    center_offset = np.array([COURT_WIDTH_FT / 2, COURT_LENGTH_FT * 0.75])
    sv_centroid = sv.Color.from_rgb_tuple(centroid_color)
    
    # Offense centroids
    for traj in cluster.centroid_offense:
        if len(traj) > 1:
            path = traj + center_offset
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=sv_centroid,
                court=court,
                thickness=centroid_thickness
            )
    
    # Defense centroids
    for traj in cluster.centroid_defense:
        if len(traj) > 1:
            path = traj + center_offset
            court = draw_paths_on_court(
                config=config,
                paths=[path],
                color=sv_centroid,
                court=court,
                thickness=centroid_thickness
            )
    
    # Ball centroid
    if cluster.centroid_ball is not None and len(cluster.centroid_ball) > 1:
        ball_path = cluster.centroid_ball + center_offset
        court = draw_paths_on_court(
            config=config,
            paths=[ball_path],
            color=sv.Color.from_rgb_tuple(BALL_COLOR),
            court=court,
            thickness=centroid_thickness
        )
    
    return court


def render_possession_video(
    possession: Possession,
    output_path: Path,
    fps: float = 30.0,
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> None:
    """
    Render a possession as a video file.
    
    Args:
        possession: Possession object
        output_path: Output video path
        fps: Frames per second
        team_colors: Team color mapping
    """
    if not HAS_VISUALIZATION_LIBS:
        raise ImportError("Visualization requires 'supervision' and 'sports' libraries")
    
    config = _get_court_config()
    court = draw_court(config=config)
    h, w = court.shape[:2]
    
    video_info = sv.VideoInfo(width=w, height=h, fps=fps)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sv.VideoSink(str(output_path), video_info) as sink:
        for frame_idx in range(possession.duration_frames):
            frame = draw_possession_frame(
                possession,
                frame_idx,
                court_image=court.copy(),
                team_colors=team_colors
            )
            sink.write_frame(frame)
    
    print(f"Saved possession video to {output_path}")


def get_video_clip_info(
    possession: Possession,
    source_video_path: Path,
    fps: float
) -> Dict:
    """
    Get information for extracting the source video clip for a possession.
    
    Args:
        possession: Possession object
        source_video_path: Path to the original game video
        fps: Video FPS
        
    Returns:
        Dict with clip extraction info
    """
    start_time = possession.start_frame / fps
    end_time = possession.end_frame / fps
    
    return {
        'source_video': str(source_video_path),
        'start_frame': possession.start_frame,
        'end_frame': possession.end_frame,
        'start_time_seconds': start_time,
        'end_time_seconds': end_time,
        'duration_seconds': end_time - start_time,
        'ffmpeg_command': (
            f'ffmpeg -i "{source_video_path}" '
            f'-ss {start_time:.3f} -to {end_time:.3f} '
            f'-c copy "possession_{possession.possession_id}.mp4"'
        )
    }


def save_cluster_visualizations(
    clusters: List[PlayCluster],
    possessions: List[NormalizedPossession],
    output_dir: Path,
    top_n: Optional[int] = None
) -> List[Path]:
    """
    Save visualization images for top clusters.
    
    Args:
        clusters: List of PlayCluster objects (assumed sorted by size)
        possessions: List of all NormalizedPossession objects
        output_dir: Output directory
        top_n: Number of top clusters to visualize (None = all)
        
    Returns:
        List of saved image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    clusters_to_viz = clusters[:top_n] if top_n else clusters
    
    for cluster in clusters_to_viz:
        img = draw_cluster_summary(cluster, possessions)
        
        # Add text annotation
        text = f"Play #{cluster.cluster_id} - {cluster.size} possessions"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        output_path = output_dir / f"play_{cluster.cluster_id:03d}.png"
        cv2.imwrite(str(output_path), img)
        saved_paths.append(output_path)
        print(f"Saved {output_path}")
    
    return saved_paths
