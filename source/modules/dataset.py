import random
from datasets import concatenate_datasets, Features, Sequence, load_from_disk, load_dataset
from collections import defaultdict
import os
import numpy as np
from datasets import concatenate_datasets, Dataset
from multiprocessing import Pool, get_context
from tqdm import tqdm
import shutil
import gc
import pyarrow as pa
import json
import hashlib

from modules.dsp import num_samples_to_duration_s, duration_s_to_num_samples, detect_highest_energy_region
from modules.utils import log_memory


# def flatten_raw_examples(raw_examples):
#     flattened = defaultdict(list)
#     for ex in raw_examples:
#         for key, val in ex.items():
#             flattened[f"raw_files_{key}"].append(val)
#     return dict(flattened)

def flatten_raw_examples(raw_examples):
    flattened = defaultdict(list)

    for ex in raw_examples:
        for key, val in ex.items():
            if isinstance(val, dict):
                # struct-of-lists
                for k, v in val.items():
                    flattened[f"raw_files_{key}_{k}"].append(v)
            else:
                flattened[f"raw_files_{key}"].append(val)

    return dict(flattened)


def balance_dataset_by_species(dataset, method="undersample", seed=42, max_per_file=None, min_samples_per_species=None):
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
    
    if len(dataset) == 0:
        print("An empty dataset can not be balanced.")
        return dataset

    random.seed(seed)

    # Group indices by species
    species_to_indices = {}
    for idx, name in enumerate(dataset['scientific_name']):
        species_to_indices.setdefault(name, []).append(idx)

    # Remove species with unsufficient amount of samples if min_samples_per_species is set
    species_to_remove = []
    species_removed = {}
    if min_samples_per_species:
        for key, values in species_to_indices.items():
            if len(values) < min_samples_per_species:
                species_to_remove.append(key)

    for species_key in species_to_remove:
        species_removed[species_key] = species_to_indices.pop(species_key)

    # species_removed = {key: values for key, values in species_to_indices.items() if len(values) < min_samples_per_species}
    # species_to_indices = {key: values for key, values in species_to_indices.items() if len(values) >= min_samples_per_species}

    # Get count
    counts = [len(indices) for indices in species_to_indices.values()]

    # Determine target count
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
    return dataset.select(balanced_indices), species_removed

# def flatten_features(prefix: str, features: Features) -> Features:
#     flat = {}
#     for key, value in features.items():
#         flat[f"{prefix}_{key}"] = Sequence(value)
#     return Features(flat)

# def flatten_features(prefix: str, features: Features) -> Features:
#     flat = {}

#     for key, value in features.items():
#         flat[f"{prefix}_{key}"] = Sequence(feature=value)

#     return Features(flat)


def flatten_features(prefix: str, features: Features) -> Features:
    flat = {}

    for key, value in features.items():
        flat[f"{prefix}_{key}"] = Sequence(feature=value)

    return Features(flat)

def filter_dataset_by_audio_array_length(dataset, min_duration_in_s):
    return [
        example for example in dataset
        if num_samples_to_duration_s(len(example['audio']['array']), example['audio']['sampling_rate']) >= min_duration_in_s
    ]

def batch_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset.select(range(i, min(i+batch_size, len(dataset))))

def iterable_batch_generator(iterable_dataset, batch_size):
    for batch_dict in iterable_dataset.iter(batch_size=batch_size):
        yield batch_dict

def clean_dir(temp_dir):
    if os.path.exists(temp_dir):
        print(f"Cleaning existing path: {temp_dir}")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

def create_schema_from_features(features):
    return Dataset.from_dict({name: [] for name in features}, features=features).data.table.schema

# def create_schema_from_features(features):
#     dummy = {}

#     for name, feat in features.items():
#         if isinstance(feat, Sequence):
#             dummy[name] = []
#         elif isinstance(feat, dict):
#             dummy[name] = {
#                 k: [] if isinstance(v, Sequence) else 0
#                 for k, v in feat.items()
#             }
#         else:
#             dummy[name] = 0

#     return Dataset.from_dict(
#         {k: [v] for k, v in dummy.items()},
#         features=features
#     ).data.table.schema

def create_dataset_info(temp_dir, features, total_shards):
    print("Creating dataset metadata...")

    # Load one shard to infer info
    first_ds = Dataset.from_file(os.path.join(temp_dir, f"data-00000-of-{total_shards:05d}.arrow"))

    # Count total examples across all shards
    total_examples = 0
    for i in range(total_shards):
        shard_path = os.path.join(temp_dir, f"data-{i:05d}-of-{total_shards:05d}.arrow")
        shard_ds = Dataset.from_file(shard_path)
        total_examples += len(shard_ds)

    # Create dataset_info.json
    return {
        "features": features.to_dict() if hasattr(features, 'to_dict') else features,
        "num_examples": total_examples,
        "splits": {
            "train": {
                "name": "train",
                "num_examples": total_examples#, # only used in old dataset versions
               # "shards": total_shards
            }
        }
    }

def create_state_info(total_shards, features, data_dir="tmp_process"):
    state = {
        "_data_files": [
            {"filename": f"data-{i:05d}-of-{total_shards:05d}.arrow"}
            for i in range(total_shards)
        ],
        "_fingerprint": None,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False, # True does not work with pyarrow table
        "_split": None, #"train",
    }

    # compute a dataset fingerprint
    filepaths = [f"{data_dir}/{f['filename']}" for f in state["_data_files"]]
    state["_fingerprint"] = compute_fingerprint(filepaths)

    return state


def compute_fingerprint(filepaths):
    sha = hashlib.sha1()
    for f in filepaths:
        with open(f, "rb") as fp:
            while True:
                data = fp.read(2**20)
                if not data:
                    break
                sha.update(data)
    return sha.hexdigest()

def process_batches_in_parallel(dataset, process_batch_fn, features, batch_size=10, 
                                num_workers=1, num_batches=None, initializer=None, initargs=(),
                                temp_dir="tmp_process",
                                batches_per_shard=10,
                                generate_batches_fn=None):  # Adjust based on memory
    
    # Check inputs
    if not dataset or len(dataset) == 0:
        raise ValueError("Dataset can not be empty")
    if batch_size < 1 and not generate_batches_fn:
        raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
    if num_workers < 1:
        raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")
    
    # Clean temp dir if exists
    clean_dir(temp_dir)
    
    # Create schema
    temp_schema = create_schema_from_features(features)
    print(temp_schema)
    print(features)
    
    # Init shards
    shard_idx = 0
    batches_in_current_shard = 0
    current_shard_file = os.path.join(temp_dir, f"data-{shard_idx:05d}-of-XXXXX.arrow")
    
    sink = None
    writer = None
    
    # Create multiprocessing pool
    with get_context('spawn').Pool(
        num_workers,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=10, # change this to 1 if memory starts to accumulate and there is not way to fix the memory leak
    ) as pool:
        
        try:
            # Open first shard
            sink = pa.OSFile(current_shard_file, 'wb')
            writer = pa.ipc.new_stream(sink, temp_schema)

            if generate_batches_fn is None:
                batch_gen = batch_generator(dataset, batch_size)
            else:
                batch_gen = generate_batches_fn

            # Process batches in parallel
            for batch_idx, batch_result in enumerate(tqdm(
                pool.imap_unordered(
                    process_batch_fn,
                    batch_gen,
                    chunksize=1
                ),
                total=num_batches,
                desc="Processing batches"
            )):

                # Write data from main process
                batch_table = pa.Table.from_pylist(batch_result, schema=temp_schema)
                writer.write_table(batch_table)
                batches_in_current_shard += 1
                
                # Check if we should start a new shard
                if batches_in_current_shard >= batches_per_shard:
                    # Close current shard
                    writer.close()
                    sink.close()
                    
                    # Start new shard
                    shard_idx += 1
                    batches_in_current_shard = 0
                    current_shard_file = os.path.join(temp_dir, f"data-{shard_idx:05d}-of-XXXXX.arrow")
                    
                    sink = pa.OSFile(current_shard_file, 'wb')
                    writer = pa.ipc.new_stream(sink, temp_schema)
                    
                    print(f"\nStarted new shard {shard_idx}")
                
                del batch_result, batch_table
                gc.collect()
                
                print(f"Processed batch {batch_idx}, current shard {shard_idx}")

        except Exception as e:
            safe_close(writer, sink)
            raise
        finally:
            safe_close(writer, sink)
    
    gc.collect()
    
    total_shards = shard_idx + 1
    print(f"\nCreated {total_shards} Arrow shards")
    
    # Rename files with correct total count
    for i in range(total_shards):
        old_name = os.path.join(temp_dir, f"data-{i:05d}-of-XXXXX.arrow")
        new_name = os.path.join(temp_dir, f"data-{i:05d}-of-{total_shards:05d}.arrow")
        os.rename(old_name, new_name)
    
    # Create dataset_info.json metadata
    print("Creating dataset metadata...")

    dataset_info = create_dataset_info(temp_dir, features, total_shards)
    with open(os.path.join(temp_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    state_info = create_state_info(total_shards, features, temp_dir)
    with open(os.path.join(temp_dir, "state.json"), "w") as f:
        json.dump(state_info, f, indent=2)

    print(f"Dataset with examples saved to {temp_dir}")
    print(f"Arrow files: {total_shards} shards")
    
    return temp_dir

def safe_close(writer, sink):
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
    if sink is not None:
        try:
            sink.close()
        except Exception:
            pass

def overwrite_dataset(dataset, dataset_path, store_backup=True):
    # Save to temporary location
    temp_path = dataset_path + "_temp"
    os.makedirs(temp_path, exist_ok=True)
    dataset.save_to_disk(temp_path)

    # Move old data to backup
    backup_path = dataset_path + "_backup"
    os.makedirs(backup_path, exist_ok=True)
    if os.path.exists(dataset_path):
        shutil.move(dataset_path, backup_path)

    # Move temp data into place
    shutil.move(temp_path, dataset_path)

    # Optionally remove backup
    if not store_backup:
        shutil.rmtree(backup_path)

def move_dataset(arrow_dir, dataset_path, store_backup=False):
            
    # Backup if needed
    if store_backup and os.path.exists(dataset_path):
        backup_path = dataset_path + "_backup"
        print(f"Creating backup at {backup_path}")
        shutil.copytree(dataset_path, backup_path, dirs_exist_ok=True)
    
    # Remove old dataset directory
    if os.path.exists(dataset_path):
        print(f"Removing old dataset at {dataset_path}")
        shutil.rmtree(dataset_path)
    
    # Move the arrow files directory to the target location
    print(f"Moving arrow files from {arrow_dir} to {dataset_path}")
    shutil.move(arrow_dir, dataset_path)
    
    print(f"Dataset successfully moved to {dataset_path}")
    return

def get_area_of_interest(example, max_duration_s):
    max_num_samples = int(max_duration_s * example["audio"]["sampling_rate"])
    audio_array = example["audio"]["array"]

    start_time = example['start_time']
    end_time = example['end_time']

    start_sample = duration_s_to_num_samples(start_time) if start_time else None
    end_sample = duration_s_to_num_samples(end_time) if end_time else None
    event_samples = end_sample - start_sample if start_sample and end_sample else None

    if len(audio_array) <= max_num_samples:
        return 0, len(audio_array)
    if not start_sample and not end_sample:
        return 0, max_num_samples
    elif not start_sample:
        return end_sample - max_num_samples, end_sample
    elif not end_sample:
        return start_sample, min(len(audio_array), start_sample + max_num_samples)
    elif event_samples >= max_num_samples:  
        return start_sample, min(len(audio_array), (start_sample + max_num_samples))
    elif event_samples < max_num_samples:
        shift_samples = (max_num_samples - event_samples) / 2
        return start_sample - shift_samples, min(len(audio_array), (start_sample - shift_samples + max_num_samples))
    else: 
        return 0, min(len(audio_array), max_num_samples)
    
def truncate_example(example, max_duration_s):

    # Try to get start and end time from labels
    start_time = example['start_time']
    end_time = example['end_time']
    if start_time or end_time:
        start_sample, end_sample = get_area_of_interest(example, max_duration_s)
    else:
        audio_array = example['audio']['array']
        sampling_rate = example['audio']['sampling_rate']
        start_sample, end_sample = detect_highest_energy_region(audio_array, max_duration_s, sampling_rate, hop_size=sampling_rate)


    max_num_samples = int(max_duration_s * example["audio"]["sampling_rate"])
    audio_array = example["audio"]["array"]
    example['audio']['array'] = example['audio']['array'][start_sample : end_sample]
    return example

# def truncate_example(example, max_duration_s):
#     max_num_samples = int(max_duration_s * example["audio"]["sampling_rate"])
#     audio_array = example["audio"]["array"]

#     # take area around start_time and end_time
#     start_time = example['start_time']
#     end_time = example['end_time']
#     if start_time or end_time:
#         print(start_time, end_time)

#     start_sample = duration_s_to_num_samples(start_time) if start_time else None
#     end_sample = duration_s_to_num_samples(end_time) if end_time else None
#     event_samples = end_sample - start_sample if start_sample and end_sample else None

#     # if num_samples is les than max_num_samples -> return
#     if len(audio_array) <= max_num_samples:
#         return example
#     if not start_sample and not end_sample:
#         example["audio"]["array"] = audio_array[: max_num_samples]
#     elif not start_sample:
#         example["audio"]["array"] = audio_array[(end_sample - max_num_samples) : end_sample]
#     elif not end_sample:
#         example['audio']['array'] = audio_array[start_sample : min(len(audio_array), start_sample + max_num_samples)]
#     # if start-end is more than max_duration -> [start:start+max_num_samples]
#     elif event_samples >= max_num_samples:  
#         example["audio"]["array"] = audio_array[start_sample : min(len(audio_array), (start_sample + max_num_samples))]
#     # if start-end is less than max_duration -> distribute padding around start-end
#     elif event_samples < max_num_samples:
#         shift_samples = (max_num_samples - event_samples) / 2
#         example["audio"]["array"] = audio_array[(start_sample - shift_samples) : min(len(audio_array), (start_sample - shift_samples + max_num_samples))]
#     else: 
#         example["audio"]["array"] = audio_array[: max_num_samples]
#     return example

def truncate_audio(example, max_duration_s):
    max_num_samples = int(max_duration_s * example["audio"]["sampling_rate"])
    audio_array = example["audio"]["array"]
    # take area around start_time and end_time
    start_time = example['start_time']
    end_time = example['end_time']
    duration = end_time - start_time

    # if num_samples is less than max_num_samples -> return
    if len(audio_array) <= max_num_samples:
        return example
    # if start-end is more than max_duration -> [start:start+max_num_samples]
    elif duration >= max_duration_s:
        start_sample = duration_s_to_num_samples(start_time)
        example["audio"]["array"] = example["audio"]["array"][start_sample:(start_sample + max_num_samples)]
    # if start-end is less than max_duration -> distribute padding around start-end
    elif duration < max_duration_s:
        shift_samples = duration_s_to_num_samples((max_duration_s - duration) / 2)
        start_sample = duration_s_to_num_samples(start_time)
        example["audio"]["array"] = example["audio"]["array"][(start_sample - shift_samples):(start_sample - shift_samples + max_num_samples)]

    return example

def truncate_batch(batch, max_duration_s):
    truncated_arrays = []
    for example in batch:
        example = truncate_example(example, max_duration_s)
    
    batch["audio"] = [
        {"array": arr, "sampling_rate": audio["sampling_rate"], "path": audio.get("path")}
        for arr, audio in zip(truncated_arrays, batch["audio"])
    ]

    del truncated_arrays
    return batch

# def truncate_batch(batch, max_duration_s):
#     truncated_arrays = []
#     for audio in batch["audio"]:
#         max_num_samples = int(max_duration_s * audio["sampling_rate"])
#         truncated_arrays.append(audio["array"][:max_num_samples])
    
#     batch["audio"] = [
#         {"array": arr, "sampling_rate": audio["sampling_rate"], "path": audio.get("path")}
#         for arr, audio in zip(truncated_arrays, batch["audio"])
#     ]

#     del truncated_arrays
#     return batch



    
    