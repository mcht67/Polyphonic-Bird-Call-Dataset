from datasets import load_from_disk, Sequence, Value, load_dataset
from omegaconf import OmegaConf
import time
from datetime import datetime
import json
import os

from integrations.birdnetlib.analyze import analyze_batch, init_analyzation_worker
from modules.dataset import process_batches_in_parallel, move_dataset
from modules.utils import get_num_workers

def main():

    print("Start analyzing audio with birdnetlib...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    raw_data_path = cfg.paths.raw_data
    birdnetlib_sampling_rate = cfg.analysis.sampling_rate
    analysis_metadata_path = cfg.paths.analysis_metadata

    # Load dataset with separateed sources
    raw_dataset = load_from_disk(raw_data_path) # This should work as long as dataset_info.json and state.json are created correctly

    temp_dir = "temp/analyze"
    #raw_dataset = raw_dataset.take(10)
    features = raw_dataset.features

    # Check if column sources exists, else raise error
    if not "sources" in raw_dataset.column_names:
       raise Exception("Can not analyze 'sources'. Dataset does not contain column 'sources'.")

    features = raw_dataset.features.copy()
    features['sources'] =  [{
                                "audio": {
                                    "array": Sequence(Value("float32")),
                                    "sampling_rate": Value("int64"),
                                },
                                "detections": [{
                                    "common_name": Value("string"),
                                    "scientific_name": Value("string"),
                                    "label": Value("string"),
                                    "confidence": Value("float32"),
                                    "start_time": Value("float32"),
                                    "end_time": Value("float32"),
                                }],
                            }]            

    # Multiprocessing config
    num_workers = get_num_workers(gb_per_worker=2, cpu_percentage=0.8)
    batch_size = 2 # ceil(((len(raw_dataset) + 1) / num_workers)//10)
    num_batches = (len(raw_dataset) + batch_size - 1) // batch_size
    batches_per_shard=1
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")
    
    # Calculate the start time
    start = time.time()
    print("Analyze with", num_workers, "workers and a batch size of", batch_size)

    # Write dataset with detections into temp_dir
    arrow_dir = process_batches_in_parallel(
            raw_dataset,
            process_batch_fn=analyze_batch,
            features=features,
            batch_size=batch_size,
            batches_per_shard=batches_per_shard,
            num_batches=num_batches,
            temp_dir=temp_dir,
            num_workers=num_workers,
            initializer=init_analyzation_worker,
            initargs=(birdnetlib_sampling_rate,) # has to be tuple
        )
    
    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    print("Finished analyzing audio birdnetlib!")
    print("Analysis with", num_workers, "worker took", length, "seconds!")

    # # Store stage results for dvc tracking
    data = {
        "datetime": datetime.now().isoformat(),
        "batch_size": batch_size,
        "batches_per_shard": batches_per_shard,
        "processing_time_seconds": length,
        "dataset_path": raw_data_path,
        "features": list(features.keys()),
    }

    with open(analysis_metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    # Move analyzed dataset with detections to raw dataset path
    move_dataset(arrow_dir, raw_data_path, store_backup=False)

if __name__ == '__main__':
  main()