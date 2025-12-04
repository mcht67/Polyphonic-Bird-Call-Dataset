from librosa import resample
import tempfile
import numpy as np
import subprocess
import re
import ast
from collections import Counter
import os

from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer

from modules.utils import normalize_name


def analyze_with_birdnetlib(audio_array, original_sampling_rate, birdnet_sampling_rate=48000, lat=None, lon=None):
    """
    Analyze audio array with BirdNETLib via a subprocess.

    Parameters
    ----------
    audio_array : np.ndarray
        Audio data.
    original_sampling_rate : int
        Sampling rate of the input audio.
    birdnet_sampling_rate : int, optional
        Sampling rate for BirdNETLib (default 48000).
    lat : float or str, optional
        Latitude for location filtering.
    lon : float or str, optional
        Longitude for location filtering.

    Returns
    -------
    list or None
        List of detections, or None if no detections found.
    """
    # Resample to 48 kHz
    y = resample(audio_array, orig_sr=original_sampling_rate, target_sr=birdnet_sampling_rate)

    # Save temporary .npy file
    with tempfile.NamedTemporaryFile(suffix=".npy") as tmp_file:
        np.save(tmp_file.name, y)

        # Build the command
        # TODO: remove python path and specify env instead
        cmd = [
            # "conda", "run", "-n", "birdnetlib",
            # "python", 
            "/Users/maltecohrt/miniconda3/envs/birdnetlib/bin/python",
            "source/integrations/birdnetlib/run_birdnetlib_inference.py",
            "--array_file", tmp_file.name,
            "--sr", str(birdnet_sampling_rate)
        ]

        if lat is not None:
            cmd += ["--lat", str(lat)]
        if lon is not None:
            cmd += ["--lon", str(lon)]

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Error handling for failed subprocess
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with return code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        # Ensure stdout is captured
        output = result.stdout
        if output is None:
            raise RuntimeError("No output captured from subprocess")

        # Extract list of detections
        # Extract the line containing the list of detections
        match = re.search(r"(\[\{.*?\}\])", output, re.DOTALL)
        if match:
            detections = ast.literal_eval(match.group(1))
        else:
            detections = None

        return detections

def analyze_example(example, target_sr=48000):

    # get all sources
    sources = example['sources']
    
    # analyze all sources with birdnetlib
    for source in sources:

        source_array = np.array(source['audio']['array'])
        source_sampling_rate = source['audio']['sampling_rate']

        # Resample if needed
        if source_sampling_rate != target_sr:
            print("Need to resample to", target_sr, "from", source_sampling_rate, ". This is inefficient. Resampling should be done beforehand.")
            source_array = resample(source_array,
                                orig_sr=source_sampling_rate,
                                target_sr=target_sr)

        recording = RecordingBuffer(
            _analyzer,
            buffer=source_array,          
            rate=source_sampling_rate,             
            min_conf=0.2
        )

        recording.analyze()

        source['detections'] = recording.detections

    return example

_analyzer = None

def init_analyzation_worker():
    "Start initilization of worker."
    global _analyzer
    _analyzer = Analyzer()
    "Initilization of worker succesful."

def analyze_batch(batch):
    print("Worker", os.getpid(), "Start analyzing batch")
    
    analyzed_examples = []
    for example in batch:
        analyzed_example = analyze_example(example)
        analyzed_examples.append(analyzed_example)

    print("Worker", os.getpid(), "Finished analyzing batch")
    
    return analyzed_examples
    
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