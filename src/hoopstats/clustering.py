"""
DTW-based clustering for basketball play recognition.

This module clusters similar possessions to identify common plays.
Uses Dynamic Time Warping (DTW) for trajectory comparison and
hierarchical clustering to group similar possessions.
"""
from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .models import NormalizedPossession, PlayCluster


def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    dist_func: Callable[[np.ndarray, np.ndarray], float] = None
) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    Args:
        seq1: First sequence, shape (n, d)
        seq2: Second sequence, shape (m, d)
        dist_func: Distance function for individual points. Default: Euclidean
        
    Returns:
        DTW distance (float)
    """
    if dist_func is None:
        dist_func = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
    
    n, m = len(seq1), len(seq2)
    
    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the cost matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    return dtw_matrix[n, m]


def _flatten_possession_trajectories(
    possession: NormalizedPossession,
    include_offense: bool = True,
    include_defense: bool = False,
    include_ball: bool = True
) -> np.ndarray:
    """
    Flatten possession trajectories into a single sequence for comparison.
    
    Args:
        possession: NormalizedPossession object
        include_offense: Include offensive player trajectories
        include_defense: Include defensive player trajectories
        include_ball: Include ball trajectory
        
    Returns:
        Flattened trajectory array, shape (num_timesteps, num_features)
    """
    features = []
    
    if include_ball and possession.ball_trajectory is not None:
        features.append(possession.ball_trajectory)  # (T, 2)
    
    if include_offense:
        for traj in possession.offense_trajectories:
            features.append(traj)  # Each is (T, 2)
    
    if include_defense:
        for traj in possession.defense_trajectories:
            features.append(traj)
    
    if not features:
        return np.zeros((possession.num_timesteps, 2))
    
    # Concatenate along feature dimension
    # Result: (T, 2 * num_trajectories)
    return np.hstack(features)


def compute_possession_distance(
    poss1: NormalizedPossession,
    poss2: NormalizedPossession,
    use_dtw: bool = True,
    include_offense: bool = True,
    include_defense: bool = False,
    include_ball: bool = True
) -> float:
    """
    Compute distance between two possessions.
    
    Args:
        poss1, poss2: NormalizedPossession objects
        use_dtw: If True, use DTW; otherwise use simple Euclidean
        include_offense: Include offensive trajectories
        include_defense: Include defensive trajectories
        include_ball: Include ball trajectory
        
    Returns:
        Distance (float)
    """
    seq1 = _flatten_possession_trajectories(
        poss1, include_offense, include_defense, include_ball
    )
    seq2 = _flatten_possession_trajectories(
        poss2, include_offense, include_defense, include_ball
    )
    
    if use_dtw:
        return dtw_distance(seq1, seq2)
    else:
        # Simple Euclidean distance (requires same length, which normalized does)
        return np.sqrt(np.sum((seq1 - seq2) ** 2))


def compute_distance_matrix(
    possessions: List[NormalizedPossession],
    use_dtw: bool = True,
    include_offense: bool = True,
    include_defense: bool = False,
    include_ball: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute pairwise distance matrix for all possessions.
    
    Args:
        possessions: List of NormalizedPossession objects
        use_dtw: If True, use DTW distance
        include_offense: Include offensive trajectories
        include_defense: Include defensive trajectories
        include_ball: Include ball trajectory
        verbose: Print progress
        
    Returns:
        Distance matrix, shape (n, n)
    """
    n = len(possessions)
    distances = np.zeros((n, n))
    
    total_pairs = n * (n - 1) // 2
    computed = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_possession_distance(
                possessions[i],
                possessions[j],
                use_dtw=use_dtw,
                include_offense=include_offense,
                include_defense=include_defense,
                include_ball=include_ball
            )
            distances[i, j] = dist
            distances[j, i] = dist
            
            computed += 1
            if verbose and computed % 100 == 0:
                print(f"  Computed {computed}/{total_pairs} distances...")
    
    return distances


def _compute_cluster_centroid(
    possessions: List[NormalizedPossession],
    indices: List[int]
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
    """
    Compute the centroid (average) trajectories for a cluster.
    
    Args:
        possessions: All possessions
        indices: Indices of possessions in this cluster
        
    Returns:
        Tuple of (offense_centroids, defense_centroids, ball_centroid)
    """
    cluster_poss = [possessions[i] for i in indices]
    
    if not cluster_poss:
        return [], [], None
    
    # Get max number of players across possessions
    max_offense = max(len(p.offense_trajectories) for p in cluster_poss)
    max_defense = max(len(p.defense_trajectories) for p in cluster_poss)
    num_timesteps = cluster_poss[0].num_timesteps
    
    # Average offense trajectories
    offense_centroids = []
    for player_idx in range(max_offense):
        player_trajs = []
        for poss in cluster_poss:
            if player_idx < len(poss.offense_trajectories):
                player_trajs.append(poss.offense_trajectories[player_idx])
        if player_trajs:
            offense_centroids.append(np.mean(player_trajs, axis=0))
    
    # Average defense trajectories
    defense_centroids = []
    for player_idx in range(max_defense):
        player_trajs = []
        for poss in cluster_poss:
            if player_idx < len(poss.defense_trajectories):
                player_trajs.append(poss.defense_trajectories[player_idx])
        if player_trajs:
            defense_centroids.append(np.mean(player_trajs, axis=0))
    
    # Average ball trajectory
    ball_trajs = [p.ball_trajectory for p in cluster_poss if p.ball_trajectory is not None]
    ball_centroid = np.mean(ball_trajs, axis=0) if ball_trajs else None
    
    return offense_centroids, defense_centroids, ball_centroid


def cluster_possessions(
    possessions: List[NormalizedPossession],
    distance_matrix: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 50.0,
    linkage_method: str = 'average',
    use_dtw: bool = True,
    include_offense: bool = True,
    include_defense: bool = False,
    include_ball: bool = True,
    verbose: bool = True
) -> List[PlayCluster]:
    """
    Cluster possessions into plays using hierarchical clustering.
    
    Args:
        possessions: List of NormalizedPossession objects
        distance_matrix: Pre-computed distance matrix (optional)
        n_clusters: Fixed number of clusters (optional)
        distance_threshold: Distance threshold for flat clustering (if n_clusters not set)
        linkage_method: Linkage method ('single', 'complete', 'average', 'ward')
        use_dtw: Use DTW distance if computing distance matrix
        include_offense: Include offensive trajectories in distance
        include_defense: Include defensive trajectories in distance
        include_ball: Include ball trajectory in distance
        verbose: Print progress
        
    Returns:
        List of PlayCluster objects, sorted by size (most common first)
    """
    if len(possessions) < 2:
        # Can't cluster with < 2 items
        if len(possessions) == 1:
            return [PlayCluster(
                cluster_id=0,
                possession_ids=[possessions[0].possession_id],
                size=1
            )]
        return []
    
    # Compute distance matrix if not provided
    if distance_matrix is None:
        if verbose:
            print("Computing distance matrix...")
        distance_matrix = compute_distance_matrix(
            possessions,
            use_dtw=use_dtw,
            include_offense=include_offense,
            include_defense=include_defense,
            include_ball=include_ball,
            verbose=verbose
        )
    
    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix)
    
    # Hierarchical clustering
    if verbose:
        print(f"Running hierarchical clustering (method={linkage_method})...")
    
    Z = linkage(condensed, method=linkage_method)
    
    # Get flat clusters
    if n_clusters is not None:
        labels = fcluster(Z, n_clusters, criterion='maxclust')
    else:
        labels = fcluster(Z, distance_threshold, criterion='distance')
    
    # Group possessions by cluster
    cluster_members = {}
    for i, label in enumerate(labels):
        cluster_members.setdefault(label, []).append(i)
    
    if verbose:
        print(f"Found {len(cluster_members)} clusters")
    
    # Create PlayCluster objects
    clusters = []
    for cluster_id, member_indices in cluster_members.items():
        # Get possession IDs
        poss_ids = [possessions[i].possession_id for i in member_indices]
        
        # Compute centroid trajectories
        offense_cent, defense_cent, ball_cent = _compute_cluster_centroid(
            possessions, member_indices
        )
        
        # Compute average intra-cluster distance
        if len(member_indices) > 1:
            intra_dists = []
            for i, idx_i in enumerate(member_indices):
                for idx_j in member_indices[i+1:]:
                    intra_dists.append(distance_matrix[idx_i, idx_j])
            avg_intra_dist = np.mean(intra_dists) if intra_dists else 0.0
        else:
            avg_intra_dist = 0.0
        
        cluster = PlayCluster(
            cluster_id=int(cluster_id),
            possession_ids=poss_ids,
            size=len(poss_ids),
            centroid_offense=offense_cent,
            centroid_defense=defense_cent,
            centroid_ball=ball_cent,
            avg_intra_cluster_distance=avg_intra_dist
        )
        clusters.append(cluster)
    
    # Sort by size (most common plays first)
    clusters.sort(key=lambda c: c.size, reverse=True)
    
    # Re-assign cluster IDs based on rank
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i
    
    return clusters


def get_cluster_summary(clusters: List[PlayCluster]) -> str:
    """
    Generate a text summary of clustering results.
    
    Args:
        clusters: List of PlayCluster objects
        
    Returns:
        Summary string
    """
    lines = [
        f"Play Clustering Summary",
        f"=" * 40,
        f"Total clusters: {len(clusters)}",
        f"Total possessions: {sum(c.size for c in clusters)}",
        f"",
        f"Clusters by usage:",
    ]
    
    for c in clusters:
        lines.append(
            f"  Play #{c.cluster_id}: {c.size} possessions "
            f"(avg distance: {c.avg_intra_cluster_distance:.1f})"
        )
    
    return "\n".join(lines)


def find_similar_possessions(
    target: NormalizedPossession,
    possessions: List[NormalizedPossession],
    top_k: int = 5,
    use_dtw: bool = True
) -> List[Tuple[int, float]]:
    """
    Find the most similar possessions to a target.
    
    Args:
        target: Target possession to compare against
        possessions: List of possessions to search
        top_k: Number of similar possessions to return
        use_dtw: Use DTW distance
        
    Returns:
        List of (possession_index, distance) tuples, sorted by distance
    """
    distances = []
    for i, poss in enumerate(possessions):
        if poss.possession_id == target.possession_id:
            continue
        dist = compute_possession_distance(target, poss, use_dtw=use_dtw)
        distances.append((i, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]
