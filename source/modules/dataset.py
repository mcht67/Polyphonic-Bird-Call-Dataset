import random
from datasets import concatenate_datasets, Features, Sequence
from collections import defaultdict
import os
import numpy as np
from datasets import concatenate_datasets, Dataset
from multiprocessing import Pool, get_context

from modules.dsp import num_samples_to_duration_s

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

def filter_dataset_by_audio_array_length(dataset, min_duration_in_s):
    return [
        example for example in dataset
        if num_samples_to_duration_s(len(example['audio']['array']), example['audio']['sampling_rate']) >= min_duration_in_s
    ]

def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100, remove_columns=None, batched=False, num_proc=1):
    """
    Generic batch processor for datasets.
    
    Args:
        dataset: Dataset to process.
        process_fn: Function to apply to each batch or example.
        cache_dir: Directory to store batch caches.
        prefix: Optional prefix for cache filenames.
        batch_size: Number of samples per batch.
        batched: If True, process_fn receives a list of examples (batched)
    
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
        
        batch_processed = batch_dataset.map(
            process_fn,
            cache_file_name=cache_file,
            remove_columns=remove_columns,
            batch_size=batch_size,
            batched=batched,
            num_proc=num_proc
        )
        
        processed_datasets.append(batch_processed)
    
    print(f"Concatenating {len(processed_datasets)} batches...")
    return concatenate_datasets(processed_datasets)

# def process_batches_in_parallel(dataset, process_batch_fn, batch_size=100, num_workers=1):
#     """
#     Generic batch processor for parallel processing of datasets.
    
#     Args:
#         dataset: Dataset to process.
#         process_batch_fn: Function to apply to each batch.
#         batch_size: Number of samples per batch.
    
#     Returns:
#         Concatenated processed dataset.
#     """

#     results = []

#      # Split dataset into batches
#     batches = [dataset.select(range(i,i+batch_size)) for i in range(0, len(dataset), batch_size)]

#     with Pool(num_workers) as pool:
#         results = pool.map(process_batch_fn, batches)
    
#     # Flatten list of batches
#     processed_examples = [ex for batch in results for ex in batch]
#     return processed_examples

# def process_batches_in_parallel(dataset, process_batch_fn, batch_size=100, num_workers=1):
#     """
#     Parallel batch processor. Safe for TF1 sessions.
#     """

#     # Create list of Dataset batches
#     batches = [
#         dataset.select(range(i, min(i+batch_size, len(dataset))))
#         for i in range(0, len(dataset), batch_size)
#     ]

#     import pickle

#     # Test if your function is picklable
#     try:
#         pickle.dumps(process_batch_fn)
#         print("Function is picklable!")
#     except Exception as e:
#         print(f"Function is NOT picklable: {e}")

#     with Pool(num_workers) as pool:
#         print("Entered pool.")
#         batch_results = pool.map(process_batch_fn, batches)

#     # Flatten list of lists
#     all_examples = [ex for batch in batch_results for ex in batch]

#     # Convert back to HF Dataset
#     return Dataset.from_list(all_examples)

# def process_batches_in_parallel(dataset, process_batch_fn, batch_size=100, num_workers=1, initializer=None, initargs=()):
#     batches = [
#         dataset.select(range(i, min(i+batch_size, len(dataset))))
#         for i in range(0, len(dataset), batch_size)
#     ]
    
#     # with Pool(num_workers, initializer=initializer, initargs=initargs) as pool:
#     #     batch_results = pool.map(process_batch_fn, batches)

#     with get_context('spawn').Pool(num_workers, initializer=initializer) as pool:
#         batch_results = pool.map(process_batch_fn, batches)
    
#     all_examples = [ex for batch in batch_results for ex in batch]
#     return Dataset.from_list(all_examples)

def process_batches_in_parallel(dataset, process_batch_fn, batch_size=100, 
                                 num_workers=1, initializer=None, initargs=()):
    batches = [
        dataset.select(range(i, min(i+batch_size, len(dataset))))
        for i in range(0, len(dataset), batch_size)
    ]
    
    with get_context('spawn').Pool(num_workers, initializer=initializer, 
                                     initargs=initargs) as pool:
        batch_results = pool.map(process_batch_fn, batches)
    
    all_examples = [ex for batch in batch_results for ex in batch]
    return Dataset.from_list(all_examples)


# def parallel_batch_processing(dataset, model_dir, checkpoint, batch_size=16, num_workers=4):
#     # Split dataset into batches
#     batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
#     with Pool(num_workers) as pool:
#         func = partial(process_batch, model_dir=model_dir, checkpoint=checkpoint)
#         results = pool.map(func, batches)
    
#     # Flatten list of batches
#     processed_examples = [ex for batch in results for ex in batch]
#     return processed_examples

# def process_batch(batch_examples, model_dir, checkpoint):
#     # Load a new TF session for this batch
#     session, input_node, output_node = load_separation_model(model_dir, checkpoint)
    
#     batch_processed = batch_dataset.map(
#             process_fn,
#             cache_file_name=cache_file,
#             remove_columns=remove_columns,
#             batch_size=batch_size,
#             batched=batched,
#             num_proc=num_proc
#         )
    
#     # Attach sources to examples
#     for ex, sources in zip(batch_examples, separated_batch):
#         ex['sources'] = [{"audio": {"array": np.array(src), "sampling_rate": sr}, "detections": []} 
#                          for src in sources]
    
#     session.close()
#     return batch_examples

# def process_batch(batch_examples, model_dir, checkpoint):
#     # Load a new TF session for this batch
#     session, input_node, output_node = load_separation_model(model_dir, checkpoint)
    
#     separated_batch, sr = separate_batch_stacked(
#         session, input_node, output_node,
#         [ex['audio']['array'] for ex in batch_examples],
#         sampling_rate=22050
#     )
    
#     # Attach sources to examples
#     for ex, sources in zip(batch_examples, separated_batch):
#         ex['sources'] = [{"audio": {"array": np.array(src), "sampling_rate": sr}, "detections": []} 
#                          for src in sources]
    
#     session.close()
#     return batch_examples


# def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100, remove_columns=None):
#     """
#     Generic batch processor for datasets.
    
#     Args:
#         dataset: Dataset to process.
#         process_fn: Function to apply to each batch.
#         cache_dir: Directory to store batch caches.
#         prefix: Optional prefix for cache filenames.
#         batch_size: Number of samples per batch.
    
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
#         batch_processed = batch_dataset.map(process_fn, cache_file_name=cache_file, remove_columns=remove_columns)
        
#         processed_datasets.append(batch_processed)
    
#     print(f"Concatenating {len(processed_datasets)} batches...")
#     return concatenate_datasets(processed_datasets)




    
    