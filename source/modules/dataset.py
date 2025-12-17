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

# def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100, remove_columns=None, batched=False, num_proc=1):
#     """
#     Generic batch processor for datasets.
    
#     Args:
#         dataset: Dataset to process.
#         process_fn: Function to apply to each batch or example.
#         cache_dir: Directory to store batch caches.
#         prefix: Optional prefix for cache filenames.
#         batch_size: Number of samples per batch.
#         batched: If True, process_fn receives a list of examples (batched)
    
#     Returns:
#         Concatenated processed dataset.
#     """

#     processed_datasets = []
#     total_samples = len(dataset)
    
#     for i in range(0, total_samples, batch_size):
#         end_idx = min(i + batch_size, total_samples)
#         print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}...")
        
#         batch_dataset = dataset.select(range(i, end_idx))
#         cache_file = os.path.join(cache_dir, f"{prefix}_batch_{i}_{end_idx}_cache.arrow")
        
#         batch_processed = batch_dataset.map(
#             process_fn,
#             cache_file_name=cache_file,
#             remove_columns=remove_columns,
#             batch_size=batch_size,
#             batched=batched,
#             num_proc=num_proc
#         )
        
#         processed_datasets.append(batch_processed)
    
#     print(f"Concatenating {len(processed_datasets)} batches...")
#     return concatenate_datasets(processed_datasets)

# def process_batches_in_parallel(dataset, process_batch_fn, batch_size=100, 
#                                  num_workers=1, initializer=None, initargs=()):
     
#     if not dataset or len(dataset) == 0:
#         raise ValueError("Dataset can not be empty")
    
#     if batch_size < 1:
#         raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
    
#     if num_workers < 1:
#         raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")

#     batches = [
#         dataset.select(range(i, min(i+batch_size, len(dataset))))
#         for i in range(0, len(dataset), batch_size)
#     ]

#     print("Process", len(batches), "batches with a batch size of", batch_size, "on", num_workers, "workers.")
    
#     # with get_context('spawn').Pool(num_workers, initializer=initializer, 
#     #                                  initargs=initargs) as pool:
#     #     batch_results = pool.map(process_batch_fn, batches)

#     with get_context('spawn').Pool(num_workers, initializer=initializer, 
#                                 initargs=initargs) as pool:
#         batch_results = list(tqdm(
#             pool.imap(process_batch_fn, batches),
#             total=len(batches),
#             desc="Processing batches"
#     ))
    
#     all_examples = [ex for batch in batch_results for ex in batch]
    
#     return Dataset.from_list(all_examples)

# Generator function - creates batches lazily
def batch_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset.select(range(i, min(i+batch_size, len(dataset))))

def process_batches_in_parallel(dataset, process_batch_fn, features, batch_size=10, 
                                num_workers=1, initializer=None, initargs=(),
                                temp_dir="tmp_process",
                                batches_per_shard=10):  # Adjust based on memory
    
    if not dataset or len(dataset) == 0:
        raise ValueError("Dataset can not be empty")
    if batch_size < 1:
        raise ValueError(f"batch_size has to be >= 1, actual: {batch_size}")
    if num_workers < 1:
        raise ValueError(f"num_workers has to be >= 1, actual: {num_workers}")
    
    # Clean output directory
    if os.path.exists(temp_dir):
        print(f"Cleaning existing output_path: {temp_dir}")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")
    
    # Create schema
    temp_schema = Dataset.from_dict(
        {name: [] for name in features}, 
        features=features
    ).data.table.schema
    
    # Init shards
    shard_idx = 0
    batches_in_current_shard = 0
    current_shard_file = os.path.join(temp_dir, f"data-{shard_idx:05d}-of-XXXXX.arrow")
    
    sink = None
    writer = None
    
    with get_context('spawn').Pool(
        num_workers,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=1,
    ) as pool:
        
        try:
            # Open first shard
            sink = pa.OSFile(current_shard_file, 'wb')
            writer = pa.ipc.new_stream(sink, temp_schema)
            
            for batch_idx, batch_result in enumerate(tqdm(
                pool.imap_unordered(
                    process_batch_fn,
                    batch_generator(dataset, batch_size),
                    chunksize=1
                ),
                total=num_batches,
                desc="Processing batches"
            )):
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
                
                del batch_ds, batch_result
                gc.collect()
                
                print(f"Processed batch {batch_idx}, current shard {shard_idx}")
        
        finally:
            # Close final shard
            if writer is not None:
                writer.close()
            if sink is not None:
                sink.close()
    
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
    
    # Load dataset lazily (doesn't load all data into memory!)
    # final_dataset = load_dataset(
    #     "arrow",
    #     data_files=os.path.join(temp_dir, "data-*.arrow"),
    #     streaming=True
    #    # split="train"
    # )
    
    print(f"Dataset with examples saved to {temp_dir}")
    #print(f"Arrow files: {total_shards} shards")
    
    return temp_dir

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



    
    