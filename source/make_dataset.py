import subprocess
import os

import load_datasets, separate_audio, segment_audio, balance_dataset, mix_audio

def main():

  # Load datasets
  #load_datasets.main()

  # Separate audio
  #separate_audio.main()

  # Analyze audio with birdnetlib
  # Build the command
  # TODO: remove python path and specify env instead
  cwd = os.getcwd()
  file_path = "/source/analyze_audio.py"

  analyze_cmd = [
      # "conda", "run", "-n", "birdnetlib",
      # "python", 
      "/Users/maltecohrt/miniconda3/envs/birdnetlib/bin/python",
      cwd + file_path
  ]
  print("Running:", ' '.join(analyze_cmd))
  subprocess.run(analyze_cmd)#,
            # capture_output=True,
            # text=True)
  
  # print("stdout:", result.stdout)
  # print("stderr:", result.stderr)
  # print("return code:", result.returncode)

  # Segment audio
  segment_audio.main()

  # Balance dataset
  balance_dataset.main()

  # Mix audio
  mix_audio.main()

if __name__ == '__main__':
  main()