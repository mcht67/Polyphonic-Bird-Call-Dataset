from datasets import concatenate_datasets, Features, Sequence
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
import random
from collections import defaultdict
from functools import wraps
from librosa import resample

from dsp import duration_s_to_num_samples, calculate_rms, normalize_to_dBFS, dBFS_to_gain, num_samples_to_duration_s

def with_random_state(func):
    """
    Decorator that allows a function to accept random_state parameter.
    The function can accept either a seed (int) or a state tuple.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract random_state from kwargs
        random_state = kwargs.pop('random_state', None)
        
        if random_state is None:
            # No state provided, call function normally
            return func(*args, **kwargs)
        
        # Save current state
        current_state = random.getstate()
        
        try:
            # Set the provided state
            if isinstance(random_state, int):
                # It's a seed
                random.seed(random_state)
            else:
                # It's a state tuple
                random.setstate(random_state)
            
            # Call the original function
            return func(*args, **kwargs)
        
        finally:
            # Restore original state
            random.setstate(current_state)
    
    return wrapper

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

def validate_species_tag(birdset_code, birdset_subset, birdnetlib_detection=None, scientific_name=None, common_name=None):
    """
    Takes the birdset label which is an id, the birdset subset used and a birdnetlib detection or scientific name/common name. 
    It gives back True or False depending on birdset label and birdnetlib detection or given names refer to the same species.
    If Birdnetlib detection is given, it is given priority over common name and scientific name.

    Parameters:
        birdset_code: int
        birdset_subset: str
        birdnetlibn_dection: dict

    Returns:
        True or False based on species tags matching, birdset label, birdnet detection
    """
    ebird_code, birdset_common_name, birdset_sci_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)
    birdset_label = birdset_common_name + ', ' + birdset_sci_name

    if birdnetlib_detection:
        common_name = normalize_name(birdnetlib_detection['common_name'])
        scientific_name = normalize_name(birdnetlib_detection['scientific_name'])

    if common_name == birdset_common_name and scientific_name == birdset_sci_name:
        validated = True
    else:
        validated = False

    comparision_label = common_name + ', ' + scientific_name

    return validated, birdset_label, comparision_label

def validate_species_tag_multi(birdset_codes, birdset_subset, birdnetlib_detection=None, scientific_name=None, common_name=None):
    """
    Takes in birdset codes, the birdset subset used and a birdnetlib detection or scientific name/common name. 
    It gives back True or False depending on one of the birdset codes and the birdnetlib detection or given names refering to the same species.
    If Birdnetlib detection is given, it is given priority over common name and scientific name.

    Parameters:
        birdset_codes: list of int
        birdset_subset: str
        birdnetlibn_dection: dict

    Returns:
        True or False based on species tags matching, birdset label, comparison label
    """
    if common_name: common_name = normalize_name(common_name)
    if scientific_name: scientific_name = normalize_name(scientific_name)

    for birdset_code in birdset_codes:
        ebird_code, birdset_common_name, birdset_sci_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)
        birdset_common_name = normalize_name(birdset_common_name)
        birdset_sci_name = normalize_name(birdset_sci_name)
        # birdset_label = birdset_common_name + ', ' + birdset_sci_name

        if birdnetlib_detection:
            common_name = normalize_name(birdnetlib_detection['common_name'])
            scientific_name = normalize_name(birdnetlib_detection['scientific_name'])

        is_validated_through_both_names = common_name == birdset_common_name and scientific_name == birdset_sci_name
        is_validated_through_sci_name = False
        is_validated_through_common_name = False

        if not common_name:
            common_name = 'common name not given'
            is_validated_through_sci_name = scientific_name == birdset_sci_name
        
        if not scientific_name:
            scientific_name = 'scientific name not given'
            is_validated_through_common_name = common_name == birdset_common_name

        if  is_validated_through_both_names or is_validated_through_sci_name or is_validated_through_common_name:
            validated = True
        else:
            validated = False

        comparison_label = common_name + ', ' + scientific_name

        return validated, birdset_code, comparison_label

def normalize_name(name):
    return name.strip().lower().replace(" ", "_")

def get_audio_file_name(output_dir, filename, details):
    return f"{output_dir}{Path(filename).stem}_{details}.wav"

def birdset_code_to_ebird_taxonomy(birdset_code, dataset_key):
    # Load BirdSet label mapping
    with open(f"resources/birdset_ebird_codes/{dataset_key}_ebird_codes.json") as f:
        birdset_labels = json.load(f)

    ebird_code = birdset_labels['id2label'][str(birdset_code)]

    # Load the official eBird taxonomy file
    tax = pd.read_csv("resources/ebird_taxonomy_v2024.csv")

    match = tax[tax["SPECIES_CODE"].str.lower() == ebird_code.lower()]
    if not match.empty:
        ebird_code = match.iloc[0]["SPECIES_CODE"].lower()  # e.g. 'rebunt1'
        common_name = normalize_name(match.iloc[0]["PRIMARY_COM_NAME"])
        sci_name = normalize_name(match.iloc[0]["SCI_NAME"])

    return ebird_code, common_name, sci_name

def get_most_confident_detection(detections):
    """
    Gets most confident detection for a list of detections provided by birdnetlib.

    Parameters:
        detections: list of dicts

    Returns:
        most confident detection: dict
    """
    if detections is None or len(detections) == 0:
        return None
    
    valid_indices = [i for i, d in enumerate(detections) if d is not None]
    if valid_indices:
        most_confident_detection_idx = max(valid_indices, key=lambda i: detections[i]['confidence'])
        return detections[most_confident_detection_idx]
    else:
        return None
    

def get_best_source_idx(list_of_detections_per_source, birdset_example=None, birdset_subset=None, decision_rule=None):
    """
    Takes list of detections per source and an optional decision rule. Chooses best source based in decision rule chosen.
    Per default 'highest_confidence_single_detection' is chosen as decision rule, choosing the source with the detection with highest confidence over all detectins of all sources.

    Parameters:
        list of detections per source : list of dicts
        decision rule: str

    Returns:
        index of chosen source
    """

    if decision_rule == None:
        decision_rule = 'highest_confidence_single_detection'
    
    if decision_rule == 'highest_confidence_single_detection':
        most_confident_detections = [None for i in range(len(list_of_detections_per_source))]

        for idx, detections in enumerate(list_of_detections_per_source):
            if detections:
                most_confident_detections[idx] = get_most_confident_detection(detections)

                # highest_confidence_idx = np.argmax([detection['confidence'] for detection in detections])
                # most_confident_detection[idx] = detections[highest_confidence_idx]

        best_source_idx = np.argmax([detection['confidence'] if detection is not None else -np.inf 
                                     for detection in most_confident_detections ])
        
    # if decision_rule == 'confidence_threshold_species_percentile':
        
    #     for idx, detections in enumerate(list_of_detections_per_source):
    #         if detections:
    #             # Get detections with confidence above threshold → removes uncertain detections
    #             detections = [detection for detection in detections if (detection['confidence'] > 0.9)]

    #             # Choose source when 0.1 - 0.9 percentile of detections above threshold are one species
    #             species_tags = [detection['scientific_name'] for detection in detections]
    #             scientific_name, is_dominant = check_dominant_species(detections)

    #             # check if species is in BirdSet labels
    #             validate_species_tag(birdset_code, birdset_subset, scientific_name=scientific_name)

    #              # extract all events (start, end) where detection is above threshold
    #              # and return it for later comparison with detected call bounds (compare time and species)
                
    # TODO:
    # introduce treshold
    #
    # add other decision rules:
    # - chossing source with the highest mean confidence over 5-10 highest confidence detections
    # - choosing source with highest summed confidence over 5-10 highest confidence detections
    # - choosing source with only detections of one bird
    #
    # Only use detections of the bird we are searching for?
    #
    # Use all sources with high confidence scores for one specific bird? As long as it is tagged in birdset??

    return best_source_idx 

def get_validated_sources(list_of_detections_per_source, birdset_example, birdset_subset, confidence_threshold=0.9, min_detection_percentage=0.9):

    sources = []
    for source_idx, detections in enumerate(list_of_detections_per_source):
        if detections:
            # Get detections with confidence above threshold → removes uncertain detections
            confident_detections = [detection for detection in detections if (detection['confidence'] > confidence_threshold)]

            # Choose source when 90% of detections are above threshold and refer to the same species
            detected_species = [detection['scientific_name'] for detection in confident_detections]
            dominant_species, is_dominant = check_dominant_species(detected_species, min_detection_percentage)

            if is_dominant:

                # check if species is in BirdSet labels
                birdset_species_ids = [birdset_example['ebird_code']] + birdset_example['ebird_code_multilabel']
                is_validated, birdset_code, comparison_label = validate_species_tag_multi(birdset_species_ids, birdset_subset, scientific_name=dominant_species)

                if is_validated:
                    # extract all events (start, end) where detection is above threshold
                    detection_bounds = [(detection['start_time'], detection['end_time']) for detection in confident_detections if detection['scientific_name']==dominant_species]

                    # Check if there is a source with this species already
                    same_species_indices = [idx for idx, source in enumerate(sources) if source['scientific_name'] == dominant_species]
                    if len(same_species_indices) > 1:
                        raise ValueError("Expected not more than one match")
                   
                    if not same_species_indices:
                        sources.append({'source_index': source_idx, 'birdset_code': birdset_code, 'scientific_name': dominant_species, 'detection_bounds': detection_bounds})
                    else: 
                        # Replace source if the new one has more detections
                        same_species_idx = same_species_indices[0]
                        source_with_same_species = sources[same_species_idx]
                        if len(detection_bounds) > len(source_with_same_species['detection_bounds']):
                            sources[same_species_idx] = source_with_same_species

    return sources

def check_dominant_species(detections, threshold=0.9):
    if not detections:
        return None, False

    counts = Counter(detections)
    species, count = counts.most_common(1)[0]
    ratio = count / len(detections)
    return species, ratio >= threshold

def only_target_bird_detected(detections, target_scientific_name, start_time, end_time, confidence_threshold=0.0):
    """
    Checks if only the target bird is detected within a specific time window.

    Parameters:
        detections (list[dict]): List of detection dicts with keys:
                                 'scientific_name', 'start_time', 'end_time', 'confidence'
        target_scientific_name (str): The bird to check against.
        start_time (float): Start of the time window (in seconds).
        end_time (float): End of the time window (in seconds).
        confidence_threshold (float): Minimum confidence to consider a detection valid.

    Returns:
        bool: True  -> only the target bird detected (and at least one detection exists)
              False -> if no detections or another bird with >= confidence_threshold was detected
    """
    def overlaps(det_start, det_end, win_start, win_end):
        """Return True if detection overlaps the time window."""
        return not (det_end <= win_start or det_start >= win_end)

    # Filter detections that overlap the time window
    window_detections = [
        det for det in detections
        if overlaps(det["start_time"], det["end_time"], start_time, end_time)
    ]

    # If no detections at all → return False
    if not window_detections:
        return False

    # Check each detection
    for det in window_detections:
        if det["confidence"] >= confidence_threshold:
            # If detection is not the target species → False
            if normalize_name(det["scientific_name"]) != normalize_name(target_scientific_name):
                return False

    # If we reach here, only target bird(s) were detected
    return True

def extract_relevant_bounds(segment_start_time, segment_end_time, time_freq_bounds):
    """
    Extract and adjust time-frequency bounds relevant to a given audio segment.

    Parameters:
        segment_start_time (float): Start time of the segment (in seconds).
        segment_end_time (float): End time of the segment (in seconds).
        time_freq_bounds (list[tuple]): List of tuples
                                        (start_time, end_time, low_freq, high_freq)
                                        all in reference to the original file.

    Returns:
        list[tuple]: Relevant time-frequency bounds adjusted to be relative
                     to the start of the segment.
                     Format: (adj_start_time, adj_end_time, low_freq, high_freq)
    """
    relevant_bounds = []

    for start, end, low_f, high_f in time_freq_bounds:
        # Check overlap with segment
        if end <= segment_start_time or start >= segment_end_time:
            continue  # no overlap

        # Clip to the segment window
        clipped_start = max(start, segment_start_time)
        clipped_end = min(end, segment_end_time)

        # Shift so times are relative to the segment
        relative_start = clipped_start - segment_start_time
        relative_end = clipped_end - segment_start_time

        relevant_bounds.append((relative_start, relative_end, low_f, high_f))

    return relevant_bounds

def flatten_features(prefix: str, features: Features) -> Features:
    flat = {}
    for key, value in features.items():
        flat[f"{prefix}_{key}"] = Sequence(value)
    return Features(flat)

def generate_mix_examples(raw_data, noise_data, max_polyphony_degree, segment_length_in_s, sampling_rate, random_seed=None):

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
    polyphony_map = create_index_map_from_range(range(1, max_polyphony_degree + 1), random_state=random_seed)
    polyphony_degree = pop_random_index(polyphony_map)

    # Create signal level map
    #signal_levels_map = create_index_map_from_range(range(-6,0), random_state=random_seed)

    # Create SNR map
    snr_map = create_index_map_from_range(range(-12, 12), random_state=random_seed) 

    # Create final mix level map
    mix_levels_map = create_index_map_from_range(range(-12, -6), random_state=random_seed) 

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
            # else apply random gain between -12 and 0 dBFS
            if not raw_signals:
                signal_dBFS = 0
            else:
                signal_dBFS = random.randrange(-12,0)
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

                # Check if all mix levels have been used
                if (all(mix_levels_map.values())):
                    reset_index_map(mix_levels_map)

                # Check if all snr levels have been used
                if (all(snr_map.values())):
                    reset_index_map(snr_map)

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
                noise_dBFS = pop_random_index(snr_map)

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
                mix_dBFS = pop_random_index(mix_levels_map)

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

                # Check if all polyphony degrees have been used
                if (all(polyphony_map.values())):
                    reset_index_map(polyphony_map)
                
                polyphony_degree = pop_random_index(polyphony_map)

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

@with_random_state
def create_index_map(num_indices, random_state=None):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param num_indices: Number of indices of the dict to create
            :type num_indices: int
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''

    # Handle random state if provided
    if random_state is not None:
        current_state = None
        if isinstance(random_state, int):
            # It's a seed
            current_state = random.getstate()
            random.seed(random_state)
        else:
            # It's a state tuple
            current_state = random.getstate()
            random.setstate(random_state)
    try:
        # Setup index map
        indices = list(range(num_indices))
        random.shuffle(indices)
        index_map = dict(zip(indices, [False] * len(indices)))
        return index_map
    finally:
        # Restore original state if changed
        if random_state is not None and current_state is not None:
            random.setstate(current_state)

@with_random_state
def create_index_map_from_range(range):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param range: Range of indices of the dict to create
            :type range: range
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''

    # Setup random indexing
    indices = list(range)
    random.shuffle(indices)
    index_map = dict(zip(indices, [False] * len(indices)))
    return index_map

def pop_random_index(index_map):
    '''
            Gets next false key from index map and sets it to True. 
            
            :return: Pseudo random index
            :rtype: int
    '''
    # Find the keys where the value is False
    false_keys = [key for key, value in index_map.items() if value is False]

    first_key = false_keys[0]

    index_map[first_key] = True

    return first_key

def reset_index_map(index_map):
    '''
            Resets the index map by setting all values to False
    '''
    for key, value in index_map.items():
        index_map[key] = False

def flatten_raw_examples(raw_examples):
    flattened = defaultdict(list)
    for ex in raw_examples:
        for key, val in ex.items():
            flattened[f"raw_files_{key}"].append(val)
    return dict(flattened)

import random
from datasets import Dataset

import random
from datasets import Dataset

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




    
    