import numpy as np
import shutil
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_from_disk, Sequence, Value, Features, Audio, Dataset
from librosa import resample
import time

from modules.dataset import flatten_features, concatenate_datasets, filter_dataset_by_audio_array_length, flatten_raw_examples, balance_dataset_by_species
from modules.utils import IndexMap
from modules.dsp import calculate_rms, normalize_to_dBFS, dBFS_to_gain, num_samples_to_duration_s, duration_s_to_num_samples

def generate_mix_examples(raw_data, noise_data, max_polyphony_degree, signal_levels, snr_values, mix_levels, segment_length_in_s, sampling_rate, random_seed=None):

    # Decode noise audio data
    noise_data = noise_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Get segment lenght in samples
    segment_length_in_samples = duration_s_to_num_samples(segment_length_in_s, sampling_rate)

    # Filter noise by length
    filtered_noise_data = filter_dataset_by_audio_array_length(noise_data, segment_length_in_s)
    print(f"Filtered noise dataset: kept {len(filtered_noise_data)} of {len(noise_data)} samples")

    # Create polyphony degree map and get initial value
    polyphony_degrees = list(range(1, max_polyphony_degree + 1)) 
    polyphony_map = IndexMap(polyphony_degrees, random_seed=random_seed, auto_reset=True)
    
    # Create signal level map
    #signal_levels = list(range(-12, 0))
    signal_levels_map = IndexMap(signal_levels, random_seed=random_seed, auto_reset=True)

    # Create SNR map
    #snr_values = list(range(-12,12))
    snr_map = IndexMap(snr_values, random_seed=random_seed, auto_reset=True)

    # Create final mix level map
    #mix_levels = list(range(-12,-6))
    mix_levels_map = IndexMap(mix_levels, random_seed=random_seed, auto_reset=True)

    # init containers
    raw_signals = []
    raw_data_list = []
    birdset_code_multilabel = []
    original_filenames = set([])
    
    polyphony_degree = polyphony_map.pop_random()
    mix_id = 0
    raw_data = list(raw_data)

    while raw_data:

        skipped_examples = []

        for example in raw_data:

            # Ommit using two files from the same file
            original_filename = example['original_file']
            if original_filename not in original_filenames:
                original_filenames.add(original_filename)
            else:
                skipped_examples.append(example)
                continue

            # Collect signals up to polyphony degree
            audio = example["audio"]
            raw_signal = np.array(audio["array"])  # This is a float32 NumPy array
            signal_sampling_rate = audio["sampling_rate"]

            # Resample if necessary
            raw_signal = resample(raw_signal, orig_sr=audio['sampling_rate'], target_sr=sampling_rate)

            # If signal length below chosen segment duration in seconds, skip it
            if raw_signal.size < segment_length_in_samples:
                print(raw_signal.size)
                print(f'Skipping segment due to insufficient length.')
                continue

            # If stereo, sum to mono
            if raw_signal.ndim > 1:  
                raw_signal = np.mean(raw_signal, axis=0)

            # Calculate RMS for event bounds
            event_bounds = example["time_freq_bounds"]
            event_rms = calculate_rms(raw_signal, signal_sampling_rate, event_bounds)
            example["event_rms"] = event_rms
            
            # Normalize file to 0 dBFS / RMS = 1
            normalized_signal, signal_norm_gain = normalize_to_dBFS(raw_signal, 0, event_rms)
            example["norm_gain"] = signal_norm_gain

            # Get relative volume in dBFS
            # Use first signal as reference with 0 dBFS
            # else get level from signal levels map #else apply random gain between -12 and 0 dBFS
            if not raw_signals:
                signal_dBFS = 0
            else:
                signal_dBFS = signal_levels_map.pop_random() #signal_dBFS = random.randrange(-12,0)
            example["relative_dBFS"] = signal_dBFS

            # Calculate and apply linear gain factors
            signal_gain = dBFS_to_gain(signal_dBFS)
            example["gain"] = signal_gain
            leveled_signal = signal_gain * normalized_signal

            # Check if signal is long enough and pad if not
            if len(leveled_signal) < segment_length_in_samples:
                padding_length = segment_length_in_samples - len(leveled_signal)
                leveled_signal = np.pad(leveled_signal, (0, padding_length))

            # Append Signal    
            raw_signals.append(leveled_signal)
            raw_data_list.append(example)

            birdset_code = example['birdset_code']
            birdset_code_multilabel.append(birdset_code)

            # Mix colleted signals
            if len(birdset_code_multilabel) == polyphony_degree:

                # Check if still noise files left
                if not mix_id < len(filtered_noise_data):
                    print("Used all noise files!")
                    break

                 # Get noise signal
                noise_array = filtered_noise_data[mix_id]['audio']['array']
                noise_sampling_rate = filtered_noise_data[mix_id]['audio']['sampling_rate']
                noise_signal = resample(noise_array, orig_sr=noise_sampling_rate, target_sr=sampling_rate)
                noise_file = Path(filtered_noise_data[mix_id]['filepath']).name

                # Normalize noise to 0 dBFS / RMS = 1
                noise_orig_rms = calculate_rms(noise_signal, sampling_rate)
                noise_signal, noise_norm_gain = normalize_to_dBFS(noise_signal, 0, noise_orig_rms)

                # Get relative SNR in dBFS
                noise_dBFS = snr_map.pop_random()

                # Calculate and apply linear gain factor
                noise_gain = dBFS_to_gain(noise_dBFS)
                noise_signal *= noise_gain

                # Check if noise is long enough
                if len(noise_signal) < segment_length_in_samples:
                    print(len(noise_signal))
                    print(num_samples_to_duration_s(len(noise_signal), sampling_rate))
                    raise ValueError("Noise signal is too short to be mixed properly.")
                
                # Append noise signal
                raw_signals.append(noise_signal)
                
                # Trim to segment length
                raw_signals = [a[:segment_length_in_samples] for a in raw_signals]

                # Mix (sum all waveforms)
                mixed_signal = np.sum(raw_signals, axis=0)

                # Get mix level in dBFS
                mix_orig_rms = calculate_rms(mixed_signal, sampling_rate)
                mix_dBFS = mix_levels_map.pop_random()

                # Normalize to desired dBFS prevent clipping
                mixed_signal, mix_gain = normalize_to_dBFS(mixed_signal, mix_dBFS, mix_orig_rms)

                flattened_raw = flatten_raw_examples(raw_data_list)

                mix_example = {
                    "id": str(mix_id),
                    "audio": {'array': mixed_signal.copy(), 'sampling_rate': int(sampling_rate)},
                    # "sampling_rate": int(sampling_rate),
                    "polyphony_degree": int(polyphony_degree),
                    "birdset_code_multilabel": birdset_code_multilabel[:],
                    "noise_file": noise_file,
                    "noise_orig_rms": noise_orig_rms,
                    "noise_dBFS": noise_dBFS,
                    "noise_norm_gain": noise_norm_gain,
                    "noise_gain": noise_gain,
                    "mix_orig_rms": mix_orig_rms,
                    "mix_dBFS": mix_dBFS,
                    "mix_gain": mix_gain,
                    **flattened_raw.copy()
                }
                yield mix_example

                # Reset mixing stage
                mix_id += 1

                raw_signals = []
                raw_data_list = []
                birdset_code_multilabel = []
                original_filenames = set([])
                
                polyphony_degree = polyphony_map.pop_random()

        previous_len = len(raw_data)
        raw_data = skipped_examples
        if len(raw_data) == previous_len:
            print("Warning: some examples could not be used (possibly all too short or duplicate filenames or amount of files is not enough to reach polyphony degree).")
            print(f'Files left: {[e["original_file"]for e in raw_data]}')
            print(f'polyphony degree: {polyphony_degree}')
            break

def generate_mix_batches(raw_data, noise_data, max_polyphony_degree, signal_levels, snr_values, mix_levels, segment_length_in_s, sampling_rate, batch_size=100, random_seed=None):
    batch = []
    for example in generate_mix_examples(raw_data, noise_data, max_polyphony_degree, signal_levels, snr_values, mix_levels, segment_length_in_s, sampling_rate, random_seed):
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():

    print("Start mixing audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    # balanced_data_path = cfg.paths.balanced_data
    segmented_data_path =  cfg.paths.segmented_data
    noise_data_path = cfg.paths.noise_data
    polyphonic_dataset_path = cfg.paths.polyphonic_data

    method = cfg.balance_dataset.method
    max_per_file = cfg.balance_dataset.max_per_file
    random_seed = cfg.random.random_seed

    segment_length_in_s = cfg.segmentation.segment_length_in_s
    max_polyphony_degree = cfg.mix.max_polyphony_degree

    signal_levels = cfg.mix.signal_levels
    snr_values = cfg.mix.snr_values
    mix_levels = cfg.mix.mix_levels

    # Load segmented dataset# 
    segmented_dataset = load_from_disk(segmented_data_path)

    # Balance dataset
    balanced_dataset = balance_dataset_by_species(segmented_dataset, method, seed=random_seed, max_per_file=max_per_file)

    # # Load balanced dataset
    # balanced_dataset = load_from_disk(balanced_data_path)
    sampling_rate = balanced_dataset[0]['audio']['sampling_rate']
   
    # Load no bird/noise dataset
    noise_dataset = load_from_disk(noise_data_path)
    
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

    temp_dirs = []

    balanced_dataset.cast_column("audio", Audio())

    # Calculate the start time
    start = time.time()
    print("Mix the whole dataset with 1 worker.")

    for i, batch in enumerate(generate_mix_batches(balanced_dataset, noise_dataset,
                                               max_polyphony_degree, 
                                               signal_levels, snr_values, mix_levels,
                                               segment_length_in_s, 
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

    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Mixing with 1 worker took ", length, "seconds!")

    # Save final dataset
    full_dataset.save_to_disk(polyphonic_dataset_path)
    print(f'Saved mixed dataset to {polyphonic_dataset_path}', flush=True)

    # Remove tmp files
    for d in temp_dirs:
        shutil.rmtree(d)

if __name__=="__main__":
    main()