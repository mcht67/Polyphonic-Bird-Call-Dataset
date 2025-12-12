from datasets import load_from_disk
from omegaconf import OmegaConf
import psutil
import time
from math import ceil

from integrations.birdnetlib.analyze import analyze_batch, init_analyzation_worker
from modules.dataset import process_batches_in_parallel, overwrite_dataset

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
    raw_dataset = load_from_disk(raw_data_path)

    # Check if column sources exists, else raise error
    if not "sources" in raw_dataset.column_names:
       raise Exception("Can not analyze 'sources'. Dataset does not contain column 'sources'.")

    # Store detections
    num_workers = psutil.cpu_count(logical=False) or psutil.cpu_count()
    batch_size = 10 #ceil((len(raw_dataset) + 1) / num_workers)

    # Calculate the start time
    start = time.time()
    print("Analyze with", num_workers, "workers and a batch size of", batch_size)

    analyzed_dataset = process_batches_in_parallel(
            raw_dataset,
            process_batch_fn=analyze_batch,
            batch_size=batch_size,
            num_workers=num_workers,
            initializer=init_analyzation_worker,
            initargs=(birdnetlib_sampling_rate,) # has to be tuple
        )
    
    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Analysis with", num_workers, "worker took", length, "seconds!")
    
    # Save analyzed dataset
    #analyzed_dataset.save_to_disk(birdnetlib_analyzed_data_path)
    overwrite_dataset(analyzed_dataset, raw_data_path, store_backup=False)

    print("Finished analyzing audio birdnetlib!")

    sources_subset = analyzed_dataset.select_columns(["sources"])
    sources_subset.to_json(analysis_metadata_path)

if __name__ == '__main__':
  main()