import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_from_disk, Sequence, Value, Features, Audio, Dataset
from librosa import resample
import time
import math

from modules.dataset import flatten_features, flatten_raw_examples, balance_dataset_by_species, process_batches_in_parallel, move_dataset
from modules.utils import IndexMap, get_num_workers
from modules.dsp import calculate_rms, normalize_to_dBFS, dBFS_to_gain, num_samples_to_duration_s, duration_s_to_num_samples, encode_audio_array

def generate_mix_examples(raw_data, noise_data, max_polyphony_degree, signal_levels, snr_values, mix_levels, segment_length_in_s, sampling_rate, random_seed=None):

    # Get segment lenght in samples
    segment_length_in_samples = duration_s_to_num_samples(segment_length_in_s, sampling_rate)

    # Create polyphony degree map and get initial value
    polyphony_degrees = list(range(1, max_polyphony_degree + 1)) 
    polyphony_map = IndexMap(polyphony_degrees, random_seed=random_seed, auto_reset=True)
    
    # Create signal level map
    signal_levels_map = IndexMap(signal_levels, random_seed=random_seed, auto_reset=True)

    # Create SNR map
    snr_map = IndexMap(snr_values, random_seed=random_seed, auto_reset=True)

    # Create final mix level map
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
                if not mix_id < len(noise_data):
                    print("Used all noise files!")
                    break

                 # Get noise signal
                noise_array = noise_data[mix_id]['audio']['array']
                noise_sampling_rate = noise_data[mix_id]['audio']['sampling_rate']
                noise_signal = resample(noise_array, orig_sr=noise_sampling_rate, target_sr=sampling_rate)
                noise_file = Path(noise_data[mix_id]['filepath']).name

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

                audio_bytes = encode_audio_array(mixed_signal, int(sampling_rate), quality=6)

                mix_example = {
                    "id": str(mix_id),
                    "audio": {'array': mixed_signal.copy(), 'sampling_rate': int(sampling_rate)},
                    "audio": {'path': None, 'bytes': audio_bytes},
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

                # Reset mixing stage
                mix_id += 1

                raw_signals = []
                raw_data_list = []
                birdset_code_multilabel = []
                original_filenames = set([])
                
                polyphony_degree = polyphony_map.pop_random()

                yield mix_example        

        previous_len = len(raw_data)
        raw_data = skipped_examples # TODO this result in empty raw_data if no skipped files and removes all other files from raw
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

def mix_batch_generator(raw_dataset, noise_dataset, raw_data_batch_size, max_polyphony_degree):

    # Calculate max number of noise samples needed per batch
    num_raw_examples_per_polyphony_map = math.ceil((max_polyphony_degree * (max_polyphony_degree + 1)) / 2) # Gauß sum
    #average_num_files_per_mixture = number_of_files_per_polyphony_map_full_iteration / max_polyphony_degree
    num_polyphony_map_per_batch = math.ceil(raw_data_batch_size / num_raw_examples_per_polyphony_map)
    num_noise_samples_per_polyphony_map = max_polyphony_degree
    noise_batch_size = num_polyphony_map_per_batch * num_noise_samples_per_polyphony_map
    noise_data_idx = 0

    for raw_data_idx in range(0, len(raw_dataset), raw_data_batch_size):
        raw_batch = raw_dataset.select(range(raw_data_idx, min(raw_data_idx+raw_data_batch_size, len(raw_dataset))))

        if len(noise_dataset) < (noise_data_idx + noise_batch_size):
            noise_data_idx = 0
            print("Warning: noise data is not sufficient. files have to be used multiple times.")
        noise_batch = noise_dataset.select(range(noise_data_idx, noise_data_idx+noise_batch_size))
        noise_data_idx += noise_batch_size

        yield raw_batch, noise_batch

def init_mixing_worker(config):
    print("Start initilization of worker.")
    global _max_polyphony_degree, _signal_levels, _snr_values, _mix_levels, _segment_length_in_s, _sampling_rate

    _max_polyphony_degree=config['max_polyphony_degree']
    _signal_levels=config['signal_levels']
    _snr_values=config['snr_values']
    _mix_levels=config['mix_levels']
    _segment_length_in_s=config['segment_length_in_s']
    _sampling_rate=config['sampling_rate']
    print("Initilization of worker succesful.")


def mix_batch(batch):

    raw_data, noise_data = batch

    mixed_examples = []
    for example in generate_mix_examples(raw_data, noise_data, _max_polyphony_degree, _signal_levels, _snr_values, _mix_levels, _segment_length_in_s, _sampling_rate):
        mixed_examples.append(example)
    return mixed_examples


def main():

    print("Start mixing audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
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
    sampling_rate = segmented_dataset[0]['audio']['sampling_rate']
    temp_dir = "temp/mix"

    # Balance dataset
    balanced_dataset, species_removed = balance_dataset_by_species(segmented_dataset, method, seed=random_seed, max_per_file=max_per_file, min_samples_per_species=50)

    # Load no bird/noise dataset
    noise_dataset = load_from_disk(noise_data_path)
    noise_dataset = noise_dataset.cast_column("audio", Audio())

    # Filter noise by length
    filtered_noise_dataset = noise_dataset.filter(lambda example: segment_length_in_s <= num_samples_to_duration_s(len(example['audio']['array']), example['audio']['sampling_rate']))
    print(f"Filtered noise dataset: kept {len(filtered_noise_dataset)} of {len(noise_dataset)} samples")
    
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
        "audio": Audio(sampling_rate=22050, mono=True, decode=False), #{'array': Sequence(Value("float32")) , 'sampling_rate': Value("int32")},
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

    # Shuffle dataset
    shuffled_dataset = balanced_dataset.shuffle(seed=random_seed)

    # Multiprocessing configuration
    num_workers = get_num_workers(gb_per_worker=2, cpu_percentage=0.8)
    batch_size = 100 # ceil(((len(raw_dataset) + 1) / num_workers)//10)
    num_batches = (len(shuffled_dataset) + batch_size - 1) // batch_size
    batches_per_shard = 2
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")
    
    # Calculate the start time
    start = time.time()
    
    # Get mix configuration
    mix_config = {
        'max_polyphony_degree': max_polyphony_degree,
        'signal_levels': signal_levels,
        'snr_values': snr_values,
        'mix_levels': mix_levels,
        'segment_length_in_s': segment_length_in_s,
        'sampling_rate': sampling_rate,
    }

    # Setup batch generator
    batch_generator_fn = mix_batch_generator(shuffled_dataset, filtered_noise_dataset, batch_size, max_polyphony_degree)

    arrow_dir = process_batches_in_parallel(shuffled_dataset,
                                            process_batch_fn=mix_batch,
                                            features=mix_features,
                                            num_workers=num_workers,
                                            num_batches=num_batches,
                                            batches_per_shard=batches_per_shard,
                                            initializer=init_mixing_worker,
                                            initargs=(mix_config,),
                                            temp_dir=temp_dir,
                                            generate_batches_fn=batch_generator_fn
                                            )

    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Mixing with", num_workers, "workers and batch size of", batch_size, "took", length, "seconds!")

    # TODO: write mix config and removed species into dataset info

    # # Save final dataset
    move_dataset(arrow_dir, polyphonic_dataset_path, store_backup=False)

if __name__=="__main__":
    main()