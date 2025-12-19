from datasets import load_from_disk
from omegaconf import OmegaConf

print("Start dataset upload...")

# Load the parameters from the config file
cfg = OmegaConf.load("params.yaml")
segmented_data_path = cfg.paths.segmented_data
polyphonic_data_path = cfg.paths.polyphonic_data
subset = cfg.dataset.subset
huggingface_user = cfg.upload.huggingface_user

separated_dataset = load_from_disk(segmented_data_path)
separated_dataset.push_to_hub(f"{huggingface_user}/polyphonic_bird_calls", private=True)

# polyphonic_dataset = load_from_disk(polyphonic_data_path)
# polyphonic_dataset.push_to_hub(f"{huggingface_user}/polyphonic_bird_calls", private=True) #/{subset}

print("Completed dataset upload.")