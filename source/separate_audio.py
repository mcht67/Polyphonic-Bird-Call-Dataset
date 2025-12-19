from datasets import load_from_disk, Audio, Features, Sequence, Value, disable_caching
from omegaconf import OmegaConf
import time
from datetime import datetime
import json

from integrations.bird_mixit.source_separation import separate_batch, init_separation_worker
from modules.dataset import process_batches_in_parallel, move_dataset
from modules.utils import get_num_workers
from modules.dataset import truncate_batch

def add_sources_column(dataset):
   # Define the schema for the new column holding the sources data
        source_features = Features({
            "sources": Sequence({
                "audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int32")}
            })
        })

        # Update dataset schema
        empty_sources = [[]] * len(dataset)
        return dataset.add_column("sources", empty_sources, feature=source_features["sources"])


def main():

    print("Start preparing dataset for separation...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    raw_data_path = cfg.paths.raw_data
    source_separation_metadata_path = cfg.paths.source_separation_metadata

    model_dir = cfg.source_separation.model_dir
    checkpoint = cfg.source_separation.checkpoint
    sampling_rate = cfg.source_separation.sampling_rate

    max_duration_s = cfg.source_separation.max_duration_s
    cpu_percentage = cfg.source_separation.cpu_percentage
    gb_per_worker = cfg.source_separation.gb_per_worker
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path) # This does not cache the data, but loads directly from disk
    #raw_dataset = raw_dataset.select(range(8))
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Remove known bad indices
    bad_indices = [1757]  # Add more if you find them

    raw_dataset = raw_dataset.select([i for i in range(len(raw_dataset)) if i not in bad_indices])

    print(f"Removed {len(bad_indices)} corrupted files")
    print(f"Dataset size: {len(raw_dataset)}")

    # Truncate audio arrays to 30s max
    print("Start truncating examples....")
    num_workers = get_num_workers(cpu_percentage=0.8, gb_per_worker=1.5)

    raw_dataset = raw_dataset.map(
        lambda batch: truncate_batch(batch, max_duration_s=max_duration_s),
        batched=True,
        batch_size=30,  # Adjust based on your memory
        writer_batch_size=30,
        num_proc=num_workers,
        keep_in_memory=False
    )

    print("Finished truncating examples.")

    # Remove sources column if it exists to remove all old sources and detections
    if  "sources" in raw_dataset.column_names:
        raw_dataset.remove_columns('sources')
        
    # # Get processing configuration  
    num_workers = get_num_workers(gb_per_worker=2, cpu_percentage=0.8)
    batch_size = 50 # ceil(((len(raw_dataset) + 1) / num_workers)//10)
    batches_per_shard = 1
    
    # Update features
    features = raw_dataset.features.copy()
    features['sources'] = [{"audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int64")}}]

    num_batches = (len(raw_dataset) + batch_size - 1) // batch_size
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")
    
    # Calculate the start time
    print("Start separating audio...")
    start = time.time()

    # Separate sources
    temp_dir = process_batches_in_parallel(
        raw_dataset,
        process_batch_fn=separate_batch,
        features=features,
        temp_dir="temp_sep",
        batch_size=batch_size,
        batches_per_shard=batches_per_shard,
        num_batches=num_batches,
        num_workers=num_workers,
        initializer=init_separation_worker,
        initargs=(model_dir, checkpoint)
    )

    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Separation with", num_workers, "worker took", length, "seconds!")

    # # Store stage results for dvc tracking
    data = {
        "datetime": datetime.now().isoformat(),
        "batch_size": batch_size,
        "batches_per_shard": batches_per_shard,
        "processing_time_seconds": length,
        "dataset_path": raw_data_path,
        "features": list(features.keys()),
    }

    with open(source_separation_metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    # Move separated dataset to raw dataset path
    move_dataset(temp_dir, raw_data_path, store_backup=False)

    print("Finished separating audio!")

if __name__ == '__main__':
  main()