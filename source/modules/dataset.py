import random
from datasets import concatenate_datasets, Features, Sequence, Audio
from collections import defaultdict
import os
from pathlib import Path
from librosa import resample
import numpy as np

from modules.dsp import calculate_rms, dBFS_to_gain, normalize_to_dBFS, num_samples_to_duration_s, duration_s_to_num_samples
from modules.utils import create_index_map_from_range, pop_random_index, reset_index_map, IndexMap

def flatten_raw_examples(raw_examples):
    flattened = defaultdict(list)
    for ex in raw_examples:
        for key, val in ex.items():
            flattened[f"raw_files_{key}"].append(val)
    return dict(flattened)

def balance_dataset_by_species(dataset, method="undersample", seed=42, max_per_file=None):
    """
    Balances a Hugging Face dataset so all 'scientific_name' classes have equal representation.

    Parameters:
        dataset: datasets.Dataset
            Input dataset with 'scientific_name' and optionally 'original_file' columns.
        method: str
            'undersample', 'oversample', or 'controlled_oversample'.
        seed: int
            Random seed for reproducibility.
        max_per_file: int or None
            If using 'controlled_oversample', this limits how many samples can come
            from the same 'original_file' (to avoid excessive duplication).

    Returns:
        A new balanced Hugging Face dataset.
    """
    if 'scientific_name' not in dataset.column_names:
        raise ValueError("'scientific_name' column not found in dataset.")
    if method == "controlled_oversample" and 'original_file' not in dataset.column_names:
        raise ValueError("'original_file' column required for controlled_oversample method.")

    random.seed(seed)

    # Group indices by species
    species_to_indices = {}
    for idx, name in enumerate(dataset['scientific_name']):
        species_to_indices.setdefault(name, []).append(idx)
    
    # Determine target count
    counts = [len(indices) for indices in species_to_indices.values()]
    if method == "undersample":
        target_count = min(counts)
    elif method in ["oversample", "controlled_oversample"]:
        target_count = max(counts)
    else:
        raise ValueError("method must be 'undersample', 'oversample', or 'controlled_oversample'")

    balanced_indices = []

    for species, indices in species_to_indices.items():
        if method == "undersample":
            selected = random.sample(indices, target_count)
        elif method == "oversample":
            selected = random.choices(indices, k=target_count)
        elif method == "controlled_oversample":
            # Group by file within this species
            file_to_indices = {}
            for idx in indices:
                file = dataset[idx]['original_file']
                file_to_indices.setdefault(file, []).append(idx)

            # Pool with per-file limit
            pool = []
            for idxs in file_to_indices.values():
                if max_per_file is not None:
                    # Limit per file
                    pool.extend(random.sample(idxs, min(len(idxs), max_per_file)))
                else:
                    pool.extend(idxs)

            # Oversample from limited pool
            if len(pool) < target_count:
                selected = random.choices(pool, k=target_count)
            else:
                selected = random.sample(pool, target_count)
        else:
            selected = indices  # fallback (should never happen)

        balanced_indices.extend(selected)

    random.shuffle(balanced_indices)
    return dataset.select(balanced_indices)

def flatten_features(prefix: str, features: Features) -> Features:
    flat = {}
    for key, value in features.items():
        flat[f"{prefix}_{key}"] = Sequence(value)
    return Features(flat)

def generate_mix_examples(raw_data, noise_data, max_polyphony_degree, segment_length_in_s, sampling_rate, random_seed=None):

    # Decode noise audio data
    noise_data = noise_data.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Get segment lenght in samples
    segment_length_in_samples = duration_s_to_num_samples(segment_length_in_s, sampling_rate)

    # Filter noise by length
    min_duration_s = segment_length_in_s
    filtered_noise_data = [
        n for n in noise_data
        if num_samples_to_duration_s(len(n['audio']['array']), n['audio']['sampling_rate']) >= min_duration_s
    ]

    print(f"Filtered noise dataset: kept {len(filtered_noise_data)} of {len(noise_data)} samples")


    # Create polyphony degree map and get initial value
    # polyphony_map = create_index_map_from_range(range(1, max_polyphony_degree + 1), random_state=random_seed)
    # polyphony_degree = pop_random_index(polyphony_map)
    polyphony_degrees = list(range(1, max_polyphony_degree + 1)) 
    polyphony_map = IndexMap(polyphony_degrees, random_seed=random_seed, auto_reset=True)
    polyphony_degree = polyphony_map.pop_random()
    
    # Create signal level map
    #signal_levels_map = create_index_map_from_range(range(-12,0), random_state=random_seed)
    signal_levels = list(range(-12, 0))
    signal_levels_map = IndexMap(signal_levels, random_seed=random_seed, auto_reset=True)

    # Create SNR map
    #snr_map = create_index_map_from_range(range(-12, 12), random_state=random_seed)
    snr_values = list(range(-12,12))
    snr_map = IndexMap(snr_values, random_seed=random_seed, auto_reset=True)

    # Create final mix level map
    #mix_levels_map = create_index_map_from_range(range(-12, -6), random_state=random_seed) 
    mix_levels = list(range(-12,-6))
    mix_levels_map = IndexMap(mix_levels, random_seed=random_seed, auto_reset=True)

    # init containers
    raw_signals = []
    raw_data_list = []
    birdset_code_multilabel = []
    original_filenames = set([])

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
            # else get level from signal levels #else apply random gain between -12 and 0 dBFS
            if not raw_signals:
                signal_dBFS = 0
            else:
                signal_dBFS = signal_levels_map.pop_random() #signal_dBFS = random.randrange(-12,0)
            example["relative_dBFS"] = signal_dBFS

            # Calculate and apply linear gain factors
            signal_gain = dBFS_to_gain(signal_dBFS)
            example["gain"] = signal_gain
            leveled_signal = signal_gain * normalized_signal

            # TODO: Check if signal is long enough and pad if not
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

                # # Check if all mix levels have been used
                # if (all(mix_levels_map.values())):
                #     reset_index_map(mix_levels_map)

                # # Check if all snr levels have been used
                # if (all(snr_map.values())):
                #     reset_index_map(snr_map)

                # Check if still noise files left
                if not mix_id < len(filtered_noise_data):
                    print("Used all noise files!")
                    break

                 # Get noise signal
                noise_array = filtered_noise_data[mix_id]['audio']['array']
                noise_sampling_rate = filtered_noise_data[mix_id]['audio']['sampling_rate']
                noise_signal = resample(noise_array, orig_sr=noise_sampling_rate, target_sr=sampling_rate)
                noise_file = Path(filtered_noise_data[mix_id]['filepath']).name
                # noise_signal = noise_signal * 10

                # Normalize noise to 0 dBFS / RMS = 1
                noise_orig_rms = calculate_rms(noise_signal, sampling_rate)
                noise_signal, noise_norm_gain = normalize_to_dBFS(noise_signal, 0, noise_orig_rms)

                # Get relative SNR in dBFS
                noise_dBFS = snr_map.pop_random()

                # Calculate and apply linear gain factor
                noise_gain = dBFS_to_gain(noise_dBFS)
                noise_signal *= noise_gain

                # TODO: Check if noise is long enough
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

                # TODO: Get mix level in dBFS
                mix_orig_rms = calculate_rms(mixed_signal, sampling_rate)
                mix_dBFS = mix_levels_map.pop_random()

                # TODO: Normalize to desired dBFS prevent clipping
                mixed_signal, mix_gain = normalize_to_dBFS(mixed_signal, mix_dBFS, mix_orig_rms)
                
                #mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

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

                # # Check if all polyphony degrees have been used
                # if (all(polyphony_map.values())):
                #     reset_index_map(polyphony_map)
                
                polyphony_degree = polyphony_map.pop_random()

        previous_len = len(raw_data)
        raw_data = skipped_examples
        if len(raw_data) == previous_len:
            print([e['original_file']for e in raw_data])
            print(f'polyphony degree: {polyphony_degree}')
            print("Warning: some examples could not be used (possibly all too short or duplicate filenames or amount of files is not enough to reach polyphony degree).")
            break

def generate_batches(raw_data, noise_data, max_polyphony_degree, segment_length_in_s, sampling_rate, batch_size=100, random_seed=None):
    batch = []
    for example in generate_mix_examples(raw_data, noise_data, max_polyphony_degree, segment_length_in_s, sampling_rate, random_seed):
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100, remove_columns=None):
    """
    Generic batch processor for datasets.
    
    Args:
        dataset: Dataset to process.
        process_fn: Function to apply to each batch.
        cache_dir: Directory to store batch caches.
        prefix: Optional prefix for cache filenames.
        batch_size: Number of samples per batch.
    
    Returns:
        Concatenated processed dataset.
    """
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}...")
        
        batch_dataset = dataset.select(range(i, end_idx))
        cache_file = os.path.join(cache_dir, f"{prefix}_batch_{i}_{end_idx}_cache.arrow")
        batch_processed = batch_dataset.map(process_fn, cache_file_name=cache_file, remove_columns=remove_columns)
        
        processed_datasets.append(batch_processed)
    
    print(f"Concatenating {len(processed_datasets)} batches...")
    return concatenate_datasets(processed_datasets)




    
    