import subprocess
import os

import load_datasets, separate_audio, segment_audio, balance_dataset, mix_audio

def main():

  # Load datasets
  load_datasets.main()

  # Separate audio
  separate_audio.main()

  # Analyze audio with birdnetlib
  # Build the command
  # TODO: remove python path and specify env instead
  cwd = os.getcwd()
  file_path = "/source/analyze_audio.py"

  analyze_cmd = [
      "conda", "run", "--no-capture-output", "-n", "birdnetlib",
      "python", 
      cwd + file_path
  ]
  subprocess.run(analyze_cmd)

  # Segment audio
  segment_audio.main()

  # Balance dataset
  #balance_dataset.main()

  # Mix audio
  mix_audio.main()

if __name__ == '__main__':
  main()