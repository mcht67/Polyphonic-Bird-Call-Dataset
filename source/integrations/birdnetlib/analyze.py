from librosa import resample
import tempfile
import numpy as np
import subprocess
import re
import ast
import gc

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

def analyze_example(example):

    # get all sources
    sources = example['sources']
    #sources['detections'] = [[] for _ in range(len(sources["audio"]))]

    # # init detections
    # if sources["detections"] == None:
    #     sources["detections"] = [[] for _ in range(len(sources["audio"]))]
    
    # analyze all sources with birdnetlib
    for source in sources:

        source_array = np.array(source['audio']['array'])
        source_sampling_rate = source['audio']['sampling_rate']

        # Resample if needed
        if source_sampling_rate != _birdnetlib_sampling_rate:
            source_array = resample(source_array,
                                orig_sr=source_sampling_rate,
                                target_sr=_birdnetlib_sampling_rate)

        recording = RecordingBuffer(
            _analyzer,
            buffer=source_array,          
            rate=_birdnetlib_sampling_rate,             
            min_conf=0.2
        )

        recording.analyze()

        source['detections'] = recording.detections
        
        # Clear all references in RecordingBuffer
        recording.buffer = None
        recording.ndarray = None
        recording.chunks = None
        recording.detection_list = None

        # Clean up after each source
        del recording
        del source_array

    return example

_analyzer = None
_birdnetlib_sampling_rate = int(0)

def init_analyzation_worker(birdnet_sampling_rate):
    print("Start initilization of worker.")
    global _analyzer
    global _birdnetlib_sampling_rate
    _analyzer = Analyzer()
    _birdnetlib_sampling_rate = birdnet_sampling_rate
    print("Initilization of worker succesful.")

def analyze_batch(batch):
    print("Worker", os.getpid(), "Start analyzing batch")
    
    analyzed_examples = []
    batch
    for example in batch:
        analyzed_example = analyze_example(example)
        analyzed_examples.append(analyzed_example)

        # Delete analyzed example after appending
        del analyzed_example

    # Force garbage collection
    gc.collect()

    print("Worker", os.getpid(), "Finished analyzing batch")

    return analyzed_examples
