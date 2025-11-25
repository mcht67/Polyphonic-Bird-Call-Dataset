from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
import tempfile

from integrations.birdnet.utils import analyze_example
from modules.dataset import process_in_batches

# Load the parameters from the config file
cfg = OmegaConf.load("config.yaml")
source_separated_data_path = cfg.paths.source_separated_data
birdnetlib_analyzed_data_path = cfg.paths.birdnetlib_analyzed_data

# Load dataset with separateed sources
separated_dataset = load_from_disk(source_separated_data_path)
separated_dataset_dataset = separated_dataset.cast_column("audio", Audio())

 # Store detections
with tempfile.TemporaryDirectory() as temp_cache_dir:

    analyzed_dataset = process_in_batches(
                    separated_dataset,
                    process_fn=analyze_example,
                    cache_dir=temp_cache_dir
                )
    
# Save analyzed dataset
analyzed_dataset.save_to_disk(birdnetlib_analyzed_data_path)