from typing import List, Dict, Tuple
import numpy as np
import supervision as sv
from sports.common.team import TeamClassifier as RoboflowTeamClassifier

from .tracking import TrackedObject

# Define player-related class IDs (based on Roboflow notebook)
# 3: player
# 4: player-in-possession
# 5: player-jump-shot
# 6: player-layup-dunk
# 7: player-shot-block
PLAYER_CLASS_IDS = {3, 4, 5, 6, 7}

class TeamClassifier:
    """
    Wrapper around Roboflow's sports TeamClassifier.
    Uses color/appearance to cluster players into two teams.
    """
    def __init__(self, device: str = "cpu"):
        # The notebook uses 'cuda' if available, we'll default to cpu to be safe locally
        # unless you want to auto-detect.
        self.classifier = RoboflowTeamClassifier(device=device)
        self.trained = False
        self.team_map = {}  # maps track_id -> team_id (int)

    def fit(self, crops: List[np.ndarray]):
        """
        Train the classifier on a batch of player crops.
        This should be done once per game/video using a sample of frames.
        """
        if not crops:
            raise ValueError("No crops provided for team classification training.")
            
        print(f"Training TeamClassifier on {len(crops)} crops...")
        self.classifier.fit(crops)
        self.trained = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team ID for a list of crops.
        """
        if not self.trained:
            raise RuntimeError("TeamClassifier not trained. Call fit() first.")
        return self.classifier.predict(crops)

def extract_player_crops(frame: np.ndarray, tracks: List[TrackedObject]) -> List[np.ndarray]:
    """
    Extract image crops for all tracked players in the frame.
    """
    crops = []
    for t in tracks:
        # Ensure xyxy is valid
        x1, y1, x2, y2 = t.xyxy
        # Clamp to frame
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
    return crops

def assign_teams(tracks: List[TrackedObject], frames_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
    """
    High-level function to assign team IDs to all tracks in a game segment.
    
    Strategy:
    1. Gather crops from a stride of frames to train the classifier.
       (Only use tracks with class_id in PLAYER_CLASS_IDS)
    2. Train classifier.
    3. Predict team for every track instance (if player).
    4. Vote: For each unique track_id, what is the most frequent team ID?
    5. Return a map {track_id: team_id}.
    """
    classifier = TeamClassifier(device="cpu") # Use "cuda" if GPU available
    
    # Helper to filter tracks
    def is_player(t: TrackedObject) -> bool:
        return t.class_id in PLAYER_CLASS_IDS

    # 1. Collect training crops (e.g., every 30th frame)
    training_crops = []
    sorted_frames = sorted(frames_dict.keys())
    
    # Simple stride
    stride = 30
    for i in range(0, len(sorted_frames), stride):
        frame_idx = sorted_frames[i]
        frame = frames_dict[frame_idx]
        
        # Find tracks in this frame that are PLAYERS
        frame_tracks = [t for t in tracks if t.frame_idx == frame_idx and is_player(t)]
        
        current_crops = extract_player_crops(frame, frame_tracks)
        training_crops.extend(current_crops)
    
    # Fallback if no crops found
    if not training_crops:
         if sorted_frames:
            f_idx = sorted_frames[0]
            frame_tracks = [t for t in tracks if t.frame_idx == f_idx and is_player(t)]
            training_crops = extract_player_crops(frames_dict[f_idx], frame_tracks)

    if not training_crops:
        print("Warning: No player crops found for team classification.")
        return {}

    # 2. Train
    classifier.fit(training_crops)
    
    # 3. Predict & Vote
    track_team_votes = {} # track_id -> list of team_ids
    
    # Group tracks by frame for efficiency
    tracks_by_frame = {}
    for t in tracks:
        if is_player(t):
            tracks_by_frame.setdefault(t.frame_idx, []).append(t)
        
    for frame_idx, frame_tracks in tracks_by_frame.items():
        if frame_idx not in frames_dict:
            continue
        
        frame = frames_dict[frame_idx]
        crops = extract_player_crops(frame, frame_tracks)
        
        if not crops:
            continue
            
        # Predict
        team_ids = classifier.predict(crops)
        
        for t, team_id in zip(frame_tracks, team_ids):
            track_team_votes.setdefault(t.track_id, []).append(team_id)
            
    # 4. Finalize
    final_team_map = {}
    for track_id, votes in track_team_votes.items():
        # Majority vote
        team_id = max(set(votes), key=votes.count)
        final_team_map[track_id] = team_id
        
    return final_team_map
