import subprocess

import load_datasets, separate_audio, analyze_audio, segment_audio, balance_dataset, mix_audio


def main():

  # Load datasets
  #load_datasets.main()

  # Separate audio
  #separate_audio.main()

  # Analyze audio with birdnetlib
  # Build the command
  # TODO: remove python path and specify env instead
  analyze_cmd = [
      "conda", "run", "-n", "birdnetlib",
      "python", 
      #"/Users/maltecohrt/miniconda3/envs/birdnetlib/bin/python",
      "source/analyze_audio.py"
  ]
  subprocess.run(analyze_cmd)

    
  #analyze_audio.main()

  # Segment audio
  #segment_audio.main()

  # Balance dataset
  #balance_dataset.main()

  # Mix audio
  mix_audio.main()

if __name__ == '__main__':
  main()