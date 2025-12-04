from collections import Counter

from modules.utils import normalize_name

def get_most_confident_detection(detections):
    """
    Gets most confident detection for a list of detections provided by birdnetlib.

    Parameters:
        detections: list of dicts

    Returns:
        most confident detection: dict
    """
    if detections is None or len(detections) == 0:
        return None
    
    valid_indices = [i for i, d in enumerate(detections) if d is not None]
    if valid_indices:
        most_confident_detection_idx = max(valid_indices, key=lambda i: detections[i]['confidence'])
        return detections[most_confident_detection_idx]
    else:
        return None
    
def check_dominant_species(detections, threshold=0.9):
    if not detections:
        return None, False

    counts = Counter(detections)
    species, count = counts.most_common(1)[0]
    ratio = count / len(detections)
    return species, ratio >= threshold

def only_target_bird_detected(detections, target_scientific_name, start_time, end_time, confidence_threshold=0.0):
    """
    Checks if only the target bird is detected within a specific time window.

    Parameters:
        detections (list[dict]): List of detection dicts with keys:
                                 'scientific_name', 'start_time', 'end_time', 'confidence'
        target_scientific_name (str): The bird to check against.
        start_time (float): Start of the time window (in seconds).
        end_time (float): End of the time window (in seconds).
        confidence_threshold (float): Minimum confidence to consider a detection valid.

    Returns:
        bool: True  -> only the target bird detected (and at least one detection exists)
              False -> if no detections or another bird with >= confidence_threshold was detected
    """
    def overlaps(det_start, det_end, win_start, win_end):
        """Return True if detection overlaps the time window."""
        return not (det_end <= win_start or det_start >= win_end)

    # Filter detections that overlap the time window
    window_detections = [
        det for det in detections
        if overlaps(det["start_time"], det["end_time"], start_time, end_time)
    ]

    # If no detections at all → return False
    if not window_detections:
        return False

    # Check each detection
    for det in window_detections:
        if det["confidence"] >= confidence_threshold:
            # If detection is not the target species → False
            if normalize_name(det["scientific_name"]) != normalize_name(target_scientific_name):
                return False

    # If we reach this, only target bird(s) were detected
    return True