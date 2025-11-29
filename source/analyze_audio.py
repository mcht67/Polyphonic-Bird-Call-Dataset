from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
import shutil

from integrations.birdnetlib.analyze import analyze_example
from modules.dataset import process_in_batches

def main():

    print("Start analyzing audio with birdnetlib...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    source_separated_data_path = cfg.paths.source_separated_data
    birdnetlib_analyzed_data_path = cfg.paths.birdnetlib_analyzed_data

    # Load dataset with separateed sources
    separated_dataset = load_from_disk(source_separated_data_path)

    # Store detections
     # Define cache dir
    temp_cache_dir = birdnetlib_analyzed_data_path + '_cache'

    analyzed_dataset = process_in_batches(
                    separated_dataset,
                    process_fn=analyze_example,
                    cache_dir=temp_cache_dir
                )
        
    # Save analyzed dataset
    analyzed_dataset.save_to_disk(birdnetlib_analyzed_data_path)

    print("Finished analyzing audio birdnetlib!")

    try:
        shutil.rmtree(temp_cache_dir)
    except Exception as e:
        print(f"Warning: cache cleanup failed: {e}")

if __name__ == '__main__':
  main()