from datasets import load_from_disk, Sequence, Value, load_dataset
from omegaconf import OmegaConf
import time
from datetime import datetime
import os
import json
import shutil

from integrations.birdnetlib.analyze import analyze_batch, init_analyzation_worker
from modules.dataset import process_batches_in_parallel, overwrite_dataset, move_dataset, process_batches_in_parallel_iter
from modules.utils import get_num_workers

def main():

    print("Start analyzing audio with birdnetlib...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    # source_separated_data_path = cfg.paths.source_separated_data
    # birdnetlib_analyzed_data_path = cfg.paths.birdnetlib_analyzed_data
    raw_data_path = cfg.paths.raw_data
    birdnetlib_sampling_rate = cfg.analysis.sampling_rate
    analysis_metadata_path = cfg.paths.analysis_metadata

    # Load dataset with separateed sources
    raw_dataset = load_from_disk(raw_data_path) # This should work as long as dataset_info.json and state.json exist
    # raw_dataset = load_dataset(
    #     "arrow",
    #     data_files=os.path.join(raw_data_path, "data-*.arrow"),
    #     #streaming=True,
    #     cache_dir="hf_cache",
    #    split="train"
    # ) # This copies the data into cache -> use DatasetIterable instead (Streming=True) or remove cache later

    #raw_dataset = raw_dataset.take(10) #
    raw_dataset = raw_dataset.select(range(10))
    features = raw_dataset.features

    # Check if column sources exists, else raise error
    if not "sources" in raw_dataset.column_names:
    #if not "sources" in features.keys():
       raise Exception("Can not analyze 'sources'. Dataset does not contain column 'sources'.")

    # Store detections
    # num_workers = psutil.cpu_count(logical=False) or psutil.cpu_count()
    # batch_size = 10 #ceil((len(raw_dataset) + 1) / num_workers)
    num_workers = get_num_workers(gb_per_worker=2, cpu_percentage=0.8)
    batch_size = 50 # ceil(((len(raw_dataset) + 1) / num_workers)//10)
    batches_per_shard = 1

    # Calculate the start time
    start = time.time()
    print("Analyze with", num_workers, "workers and a batch size of", batch_size)

    features = raw_dataset.features.copy()
    # features['sources'] = Sequence({
    #                                     "audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int64")},#, #Audio(),
    #                                     "detections": Sequence({
    #                                         "common_name": Value("string"),
    #                                         "scientific_name": Value("string"),
    #                                         "label": Value("string"),
    #                                         "confidence": Value("float32"),
    #                                         "start_time": Value("float32"),
    #                                         "end_time": Value("float32"),
    #                                     })
    #                                 })
   

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

    #print(features['sources'])
    # raw_dataset.cast(features)
    # print(raw_dataset.features)

    num_batches = (len(raw_dataset) + batch_size - 1) // batch_size
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")

    # analyzed_dataset
    arrow_dir = process_batches_in_parallel(
            raw_dataset,
            process_batch_fn=analyze_batch,
            features=features,
            batch_size=batch_size,
            batches_per_shard=1,
            num_batches=num_batches,
            temp_dir="temp_analyze",
            num_workers=num_workers,
            initializer=init_analyzation_worker,
            initargs=(birdnetlib_sampling_rate,) # has to be tuple
        )
    
    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    print("Finished analyzing audio birdnetlib!")

    #shutil.rmtree("hf_cache")

    # Show the results : this can be altered however you like
    print("Analysis with", num_workers, "worker took", length, "seconds!")

    # # Store stage results for dvc tracking
    data = {
        "datetime": datetime.now().isoformat(),
        #"num_examples": len(separated_dataset),
        #"num_shards": int(len(separated_dataset) / (batch_size * batches_per_shard)),  # From your function
        "batch_size": batch_size,
        "batches_per_shard": batches_per_shard,
        "processing_time_seconds": length,
        "dataset_path": raw_data_path,
        "features": list(features.keys()),
    }

    with open(analysis_metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save analyzed dataset
    move_dataset(arrow_dir, raw_data_path, store_backup=False)

if __name__ == '__main__':
  main()