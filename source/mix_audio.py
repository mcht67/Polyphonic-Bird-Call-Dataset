import shutil
import tempfile
from omegaconf import OmegaConf
from datasets import load_from_disk, Sequence, Value, Features, Audio, Dataset

from modules.dataset import generate_mix_batches, flatten_features, concatenate_datasets

def main():

    print("Start mixing audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    balanced_data_path = cfg.paths.balanced_data
    noise_data_path = cfg.paths.noise_data
    polyphonic_dataset_path = cfg.paths.polyphonic_data

    # Load balanced dataset
    balanced_dataset = load_from_disk(balanced_data_path)
    sampling_rate = balanced_dataset[0]['audio']['sampling_rate']
    # TODO: Check if necessary
    #balanced_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    print(balanced_dataset)


    # Load no bird/noise dataset
    noise_dataset = load_from_disk(noise_data_path)
    # TODO: Check if necessary
    # noise_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    print(noise_dataset)
    
    # Setup features
    raw_features = balanced_dataset.features
    flattened_raw_features = flatten_features("raw_files", raw_features)

    additional_raw_features = {
        "raw_files_event_rms": Sequence(Value("float32")),
        "raw_files_norm_gain": Sequence(Value("float32")),
        "raw_files_relative_dBFS": Sequence(Value("float32")),
        "raw_files_gain": Sequence(Value("float32"))
    }

    mix_features = Features({
        "id": Value("string"),
        "audio": {'array': Sequence(Value("float32")) , 'sampling_rate': Value("int32")},
        "polyphony_degree": Value("int32"),
        "birdset_code_multilabel": Sequence(Value("int32")),
        "noise_file": Value("string"),
        "noise_orig_rms": Value("float32"),
        "noise_dBFS": Value("float32"),
        "noise_norm_gain": Value("float32"),
        "noise_gain": Value("float32"),
        "mix_orig_rms": Value("float32"),
        "mix_dBFS": Value("float32"),
        "mix_gain": Value("float32"),
        **flattened_raw_features,
        **additional_raw_features
    })

    # Generate batches
    print("Mix audio in batches", flush=True)

    max_polyphony_degree = 6
    segment_length_in_s = 5
    random_seed = 42

    temp_dirs = []

    balanced_dataset.cast_column("audio", Audio())

    for i, batch in enumerate(generate_mix_batches(balanced_dataset, noise_dataset,
                                               max_polyphony_degree, segment_length_in_s, 
                                               sampling_rate, random_seed=random_seed)):
        ds = Dataset.from_list(batch, features=mix_features)
        print(f'finished batch {i}')
        
        tmp_dir = tempfile.mkdtemp(prefix=f"mix_batch_{i}_")
        ds.save_to_disk(tmp_dir)
        temp_dirs.append(tmp_dir)

    # Concatenate batches
    print("Concatenate batches", flush=True)
    datasets = [load_from_disk(d) for d in temp_dirs]
    full_dataset = concatenate_datasets(datasets)

    # Save final dataset
    full_dataset.save_to_disk(polyphonic_dataset_path)
    print(f'Saved mixed dataset to {polyphonic_dataset_path}', flush=True)

    # Remove tmp files
    for d in temp_dirs:
        shutil.rmtree(d)

if __name__=="__main__":
    main()