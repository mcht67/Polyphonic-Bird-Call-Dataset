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

from modules.dsp import num_samples_to_duration_s
from modules.utils import log_memory


def flatten_raw_examples(raw_examples):
    flattened = defaultdict(list)
    for ex in raw_examples:
        for key, val in ex.items():
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
    species_removed = {}
    if min_samples_per_species:
        for key, values in species_to_indices.items():
            if len(values) < min_samples_per_species:
                species_removed[key] = species_to_indices.pop(key)

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

def flatten_features(prefix: str, features: Features) -> Features:
    flat = {}
    for key, value in features.items():
        flat[f"{prefix}_{key}"] = Sequence(value)
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
     # Clean output directory
    if os.path.exists(temp_dir):
        print(f"Cleaning existing path: {temp_dir}")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

def create_schema_from_features(features):
    return Dataset.from_dict({name: [] for name in features}, features=features).data.table.schema

def create_dataset_info(temp_dir, features, total_shards):
    # After renaming files
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
                "num_examples": total_examples,
                "num_shards": total_shards
            }
        }
    }

def create_state_info(total_shards):
    # Create state.json
    return {
        "_data_files": [
            {"filename": f"data-{i:05d}-of-{total_shards:05d}.arrow"}
            for i in range(total_shards)
        ],
        "_fingerprint": None,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
    }

def process_batches_in_parallel(dataset, process_batch_fn, features, batch_size=10, 
                                num_workers=1, num_batches=None, initializer=None, initargs=(),
                                temp_dir="tmp_process",
                                batches_per_shard=10):  # Adjust based on memory
    
    # Check inputs
    if not dataset or len(dataset) == 0:
        raise ValueError("Dataset can not be empty")
    if batch_size < 1:
        raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
    if num_workers < 1:
        raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")
    
    # Clean temp dir if exists
    clean_dir(temp_dir)
    
    # Create schema
    temp_schema = create_schema_from_features(features)
    
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
            
            # Process batches in parallel
            for batch_idx, batch_result in enumerate(tqdm(
                pool.imap_unordered(
                    process_batch_fn,
                    batch_generator(dataset, batch_size),
                    chunksize=1
                ),
                total=num_batches,
                desc="Processing batches"
            )):
                # Write data from main process
                batch_table = pa.Table.from_pylist(batch_result, schema=temp_schema)
                writer.write_table(batch_table)
                # batch_ds = Dataset.from_list(batch_result, features=features)
                # writer.write_table(batch_ds.data.table)

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

    state_info = create_state_info(total_shards)
    with open(os.path.join(temp_dir, "state.json"), "w") as f:
        json.dump(state_info, f, indent=2)

    print(f"Dataset with examples saved to {temp_dir}")
    print(f"Arrow files: {total_shards} shards")
    
    return temp_dir

def process_batches_in_parallel_iter(dataset_iterable, process_batch_fn, features, batch_size=10, 
                                num_workers=1, num_batches=None, initializer=None, initargs=(),
                                temp_dir="tmp_process",
                                batches_per_shard=10):  # Adjust based on memory
    
    # Check inputs
    if hasattr(dataset_iterable, "__iter__") is False:
        raise TypeError("dataset must be iterable")
    if batch_size < 1:
        raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
    if num_workers < 1:
        raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")
    
    # Clean temp dir if exists
    clean_dir(temp_dir)
    
    # Create schema
    temp_schema = create_schema_from_features(features)
    
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
            
            # Process batches in parallel
            for batch_idx, batch_result in enumerate(tqdm(
                pool.imap_unordered(
                    process_batch_fn,
                    iterable_batch_generator(dataset_iterable, batch_size),
                    chunksize=1
                ),
                total=num_batches,
                desc="Processing batches"
            )):
            
                # Write data from main process    
                # batch_table = table_from_examples(batch_result, temp_schema)
                # writer.write_table(batch_table)
                batch_ds = Dataset.from_list(batch_result, features=features)
                writer.write_table(batch_ds.data.table)

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
                
                del batch_ds, batch_table
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

    state_info = create_state_info(total_shards)
    with open(os.path.join(temp_dir, "state.json"), "w") as f:
        json.dump(state_info, f, indent=2)

    print(f"Dataset with examples saved to {temp_dir}")
    print(f"Arrow files: {total_shards} shards")
    
    return temp_dir

def align_example_to_schema(example, schema):
    aligned = {}

    for field in schema:
        name = field.name
        value = example.get(name, None)
        field_type = field.type

        # Handle list fields
        if pa.types.is_list(field_type):
            if value is None:
                aligned[name] = []
            elif isinstance(value, list):
                # Remove None-only lists
                if all(v is None for v in value):
                    aligned[name] = []
                else:
                    aligned[name] = value
            else:
                raise TypeError(f"{name} expected list, got {type(value)}")

        # Handle scalar fields
        else:
            aligned[name] = value

    return aligned

def table_from_examples(examples, schema):
    arrays = []

    for field in schema:
        name = field.name
        field_type = field.type

        column = [ex.get(name, None) for ex in examples]

        # Handle list types
        if pa.types.is_list(field_type):
            column = [
                [] if (v is None or (isinstance(v, list) and all(x is None for x in v)))
                else v
                for v in column
            ]

        array = pa.array(column, type=field_type)
        arrays.append(array)

    return pa.Table.from_arrays(arrays, schema=schema)


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

# def process_batches_in_parallel(dataset, process_batch_fn, features, batch_size=10, 
#                                 num_workers=1, initializer=None, initargs=(),
#                                 tmp_dir="tmp_sep"):

#     if not dataset or len(dataset) == 0:
#         raise ValueError("Dataset can not be empty")
#     if batch_size < 1:
#         raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
#     if num_workers < 1:
#         raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")
    
#     # Clean up tmp_dir if it exists
#     if os.path.exists(tmp_dir):
#         print(f"Cleaning existing tmp_dir: {tmp_dir}")
#         shutil.rmtree(tmp_dir)

#     num_batches = (len(dataset) + batch_size - 1) // batch_size
#     print("Process", num_batches, "batches with a batch size of", batch_size,
#           "on", num_workers, "workers.")
    
#     # Create a temporary dataset to get the schema
#     temp_schema = Dataset.from_dict(
#         {name: [] for name in features}, 
#         features=features
#     ).data.table.schema
    
#     arrow_file = os.path.join(tmp_dir, "data.arrow")

#     os.makedirs(tmp_dir, exist_ok=True)
#     shard_paths = []

#     with get_context('spawn').Pool(
#         num_workers,
#         initializer=initializer,
#         initargs=initargs,
#         maxtasksperchild=1,   # or higher, e.g. 5–10
#     ) as pool:
  
#         with pa.OSFile(arrow_file, 'wb') as sink:
#             with pa.ipc.new_stream(sink, temp_schema) as writer:
                
#                 for batch_idx, batch_result in enumerate(tqdm(
#                     pool.imap_unordered(
#                         process_batch_fn,
#                         batch_generator(dataset, batch_size),
#                         chunksize=1
#                     ),
#                     total=num_batches,
#                     desc="Processing batches"
#                 )):
#                     # Convert batch to Arrow table and write directly
#                     batch_ds = Dataset.from_list(batch_result, features=features)
#                     writer.write_table(batch_ds.data.table)
                    
#                     del batch_ds, batch_result
#                     gc.collect()

#                     print(f"Processed batch {batch_idx}")
        
#     gc.collect()
    
#      # Load using stream reader
#     print("Finalizing dataset...")
#     with pa.memory_map(arrow_file, 'r') as source:
#         with pa.ipc.open_stream(source) as reader:  # ← Changed from open_file
#             arrow_table = reader.read_all()
    
#     # Create Dataset from the table
#     final_dataset = Dataset(arrow_table)
     
#     return final_dataset

# def process_batches_in_parallel(dataset, process_batch_fn, batch_size=10, 
#                                  num_workers=1, initializer=None, initargs=()):
    
#     if not dataset or len(dataset) == 0:
#         raise ValueError("Dataset can not be empty")
#     if batch_size < 1:
#         raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
#     if num_workers < 1:
#         raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")

#     num_batches = (len(dataset) + batch_size - 1) // batch_size
#     print("Process", num_batches, "batches with a batch size of", batch_size, "on", num_workers, "workers.")
    
#     all_examples = []
    
#     with get_context('spawn').Pool(num_workers, initializer=initializer, 
#                                      initargs=initargs) as pool: #,maxtasksperchild=1
#         # Process batches with imap (lazy) instead of map (eager)
#         for batch_result in tqdm(
#             pool.imap(process_batch_fn, batch_generator(dataset, batch_size)),
#             total=num_batches,
#             desc="Processing batches"
#         ):
#             # Flatten and append this batch's results
#             all_examples.extend(batch_result)
            
#             # Clean up batch result immediately
#             del batch_result
            
#             # Force GC every few batches
#             if len(all_examples) % (batch_size * 5) < batch_size:
#                 gc.collect()

#             print("Finished batch processing.")
#             log_memory()
    
#     # Final cleanup
#     gc.collect()
    
#     return Dataset.from_list(all_examples)

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

def truncate_example(example, max_duration_s):
    max_num_samples = int(max_duration_s * example["audio"]["sampling_rate"])
    example["audio"]["array"] = example["audio"]["array"][:max_num_samples]
    return example

def truncate_batch(batch, max_duration_s):
    truncated_arrays = []
    for audio in batch["audio"]:
        max_num_samples = int(max_duration_s * audio["sampling_rate"])
        truncated_arrays.append(audio["array"][:max_num_samples])
    
    batch["audio"] = [
        {"array": arr, "sampling_rate": audio["sampling_rate"], "path": audio.get("path")}
        for arr, audio in zip(truncated_arrays, batch["audio"])
    ]

    del truncated_arrays
    return batch



    
    