from datasets import load_dataset
from omegaconf import OmegaConf
import random

def is_no_bird(example):
        return example['ebird_code'] is None and example['ebird_code_multilabel'] == [] and example['ebird_code_secondary'] is None

def separate_to_noise_and_test_split(soundscape_dataset):

     # Step 1: Create a boolean mask for "no bird" examples
    no_bird_mask = [is_no_bird(ex) for ex in soundscape_dataset]

    # Step 2: Get indices of "no bird" examples
    no_bird_indices = [i for i, flag in enumerate(no_bird_mask) if flag]

    # Step 3: Randomly select half of those 
    n = len(no_bird_indices)
    selected_no_bird_indices = set(random.sample(no_bird_indices, n // 2))

    # Step 4: Split dataset
    # - noise_dataset: subset with selected "no bird" examples
    # - soundscape_dataset_filtered: dataset with those removed
    noise_dataset = soundscape_dataset.select(list(selected_no_bird_indices))
    soundscape_dataset_filtered = soundscape_dataset.filter(
        lambda _, idx: idx not in selected_no_bird_indices,
        with_indices=True
    )

    return noise_dataset, soundscape_dataset_filtered

def main():

    print("Start loading datasets...")
    
    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    dataset_subset = cfg.dataset.subset
    raw_data_path = cfg.paths.raw_data
    test_data_path = cfg.paths.test_data
    noise_data_path = cfg.paths.noise_data

    # Load Xeno Canto data
    xc_subset_name = dataset_subset + '_xc'
    raw_dataset = load_dataset('DBD-research-group/BirdSet', xc_subset_name, split='train', trust_remote_code=True)
    # TODO: Remove, only here for test runs
    raw_dataset = raw_dataset.select(range(200, 300))

    # Load soundscape data
    soundscape_subset_name = dataset_subset +'_scape'
    soundscape_5s_dataset = load_dataset('DBD-research-group/BirdSet', soundscape_subset_name, split='test_5s', trust_remote_code=True)

    # Get augmentation and test split
    noise_dataset, test_dataset = separate_to_noise_and_test_split(soundscape_5s_dataset)

    # Store datasets
    raw_dataset.save_to_disk(raw_data_path)
    test_dataset.save_to_disk(test_data_path)
    noise_dataset.save_to_disk(noise_data_path)

    print("Finished loading datasets!")

if __name__ == '__main__':
  main()