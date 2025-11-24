
import argparse
import os
os.environ.pop("MPLBACKEND", None)  # remove Jupyter backend setting

import matplotlib # necessary for BirdNet Analyzer to load properly
matplotlib.use("Agg")

from librosa import resample
import numpy as np

from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from datetime import datetime

parser = argparse.ArgumentParser(description="Run BirdNETLib on a NumPy array.")
parser.add_argument("--array_file", type=str, required=True,
                    help="Path to the .npy file containing audio array")
parser.add_argument("--sr", type=int, required=True,
                    help="Sample rate of the audio array")
parser.add_argument("--min_conf", type=float, default=0.75,
                    help="Minimum confidence for detections")
parser.add_argument("--lat", type=str, default=None,
                    help="Latitude of recording for location filtering.")
parser.add_argument("--lon", type=str, default=None,
                    help="Longitude of recording for location filtering.")
args = parser.parse_args()

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

audio_array = np.load(args.array_file)

recording = RecordingBuffer(
    analyzer,
    lat=args.lat,
    lon=args.lon,
    buffer=audio_array,          
    rate=args.sr,              
    min_conf=0.2
)

recording.analyze()
print(recording.detections)
