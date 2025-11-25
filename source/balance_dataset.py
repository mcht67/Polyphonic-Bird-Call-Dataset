from omegaconf import OmegaConf
from datasets import load_from_disk

from modules.dataset import balance_dataset_by_species

def main():

    print("Start balancing dataset...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    segmented_data_path = cfg.paths.segmented_data
    balanced_data_path = cfg.paths.balanced_data

    # Load segmented data
    segmented_dataset = load_from_disk(segmented_data_path)

    # Balance data by species
    balanced_dataset = balance_dataset_by_species(segmented_dataset)

    # Store balanced dataset
    balanced_dataset.save_to_disk(balanced_data_path)

    print("Finished balancing dataset!")

if __name__ == '__main__':
  main()