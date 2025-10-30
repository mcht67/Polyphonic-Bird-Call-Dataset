
from datasets import Audio, load_dataset
import datasets
from functools import partial
import tempfile
from pathlib import Path
import numpy as np

from source_separation import load_separation_model, separate_audio, separate_batch, separate_batch_stacked
from utils import process_in_batches

import time


def separate_example(example, birdset_subset, separation_session_data=None):

    # get audio, filename and ebird-code
    audio = example["audio"] # acces audio_array by audio['array'] and sampling_rate by audio['sampling_rate']
    filename = Path(audio["path"]).name
    ebird_code = example["ebird_code"]

    # do source separation
    session, input_node, output_node = separation_session_data
    sources, source_sr = separate_audio(session,
                   input_node,
                   output_node,
                   audio['array'],
                   input_sampling_rate=audio['sampling_rate']
                   )
    
    for i, src in enumerate(sources):
        example[f"source_{i}"] = src

    return example

def extract_audio_batch(dataset, batch_indices):
    batch = dataset.select(batch_indices)

    arrays = [ex["audio"]["array"] for ex in batch]
    srs    = [ex["audio"]["sampling_rate"] for ex in batch]
    origs  = [ex["original_index"] for ex in batch]

    return arrays, srs, origs




# ------------------------
# Load Dataset
# ------------------------


# Load from BirdSet
raw_dataset = load_dataset('DBD-research-group/BirdSet', 'HSN_xc', split='train', trust_remote_code=True)
raw_dataset = raw_dataset.cast_column("audio", Audio()) # original sampling rate of 32kHz is preserved

# TODO: remove subset
subset_indices = range(0, 5)
subset = raw_dataset.select(subset_indices)

# --------------------------------
# Separate each example separately
# --------------------------------
# Elapsed time single example processing: 208.620341 seconds
print("Start single example separation")

# Save the current cache path
original_cache = datasets.config.HF_DATASETS_CACHE

start = time.perf_counter()

# Load source separation model
session, input_node, output_node = load_separation_model(model_dir="resources/bird_mixit_model_checkpoints/output_sources4", 
                        checkpoint="resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
separation_session_data = (session, input_node, output_node)

# Separate each example
with tempfile.TemporaryDirectory() as temp_cache_dir:
    preprocess_fn = partial(
        separate_example,
        birdset_subset = 'HSN', 
        separation_session_data=separation_session_data
    )

    modified_dataset = process_in_batches(
                    subset,
                    process_fn=preprocess_fn,
                    cache_dir=temp_cache_dir,
                )
    
end = time.perf_counter()
print(f"Elapsed time single example processing: {end - start:.6f} seconds")

# After the block ends, restore it
datasets.config.HF_DATASETS_CACHE = original_cache

# --------------------------------
# Separate each examples batchwise
# --------------------------------
print("Start batchwise separation")
start = time.perf_counter()

subset = subset.add_column("original_index", list(range(len(subset))))

arrays, sampling_rates, orig_indices = extract_audio_batch(subset, subset_indices)

# Load source separation model
session, input_node, output_node = load_separation_model(model_dir="resources/bird_mixit_model_checkpoints/output_sources4", 
                        checkpoint="resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
separation_session_data = (session, input_node, output_node)

# Separate examples batchwise
sources, source_sampling_rate = separate_batch(session, input_node=input_node, output_node=output_node, audio_list=arrays, input_sampling_rate=sampling_rates[0])

# Create result buffers aligned with original dataset
num_sources = sources[0].shape[0]

source_columns = {
    "sources_sampling_rate": [None] * len(subset),
    **{f"source_{s}": [None] * len(subset) for s in range(num_sources)}
}

# Fill results per sample
for tmp_i, orig_i in enumerate(orig_indices):
    source_columns['sources_sampling_rate'][orig_i] = source_sampling_rate
    for s in range(num_sources):
        source_columns[f"source_{s}"][orig_i] = sources[tmp_i][s]

# Add all source columns efficiently
for name, col in source_columns.items():
    subset = subset.add_column(name, col)

# for tmp_index, orig_index in enumerate(orig_indices):
#     example = subset.select(orig_index)

#     for source_index, source_array in enumerate(sources[tmp_index]):
#         example[f"source_{source_index}"] = source_array

# # Later do this to only use add_column() once
# all_batches = []
# for batch_indices in ...:
#     batch_result = process_batch(...)
#     all_batches.append(batch_result)

# When all done, merge or write to disk once


end = time.perf_counter()
print(f"Elapsed time batchwise processing: {end - start:.6f} seconds")

# ----------------------------------------
# Separate each stacked examples batchwise
# ----------------------------------------