from datasets import load_from_disk
from omegaconf import OmegaConf
import psutil
import time

from integrations.birdnetlib.analyze import analyze_batch, init_analyzation_worker
from modules.dataset import process_batches_in_parallel

def main():

    print("Start analyzing audio with birdnetlib...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    source_separated_data_path = cfg.paths.source_separated_data
    birdnetlib_analyzed_data_path = cfg.paths.birdnetlib_analyzed_data
    sampling_rate = cfg.analysis.sampling_rate

    # Load dataset with separateed sources
    separated_dataset = load_from_disk(source_separated_data_path)

    # Store detections
    num_workers = psutil.cpu_count(logical=False) or psutil.cpu_count()
    batch_size = int((len(separated_dataset) + 1) / num_workers)

    # Calculate the start time
    start = time.time()
    print("Analyze with", num_workers, "workers and a batch size of", batch_size)

    analyzed_dataset = process_batches_in_parallel(
            separated_dataset, 
            process_batch_fn=analyze_batch,
            batch_size=batch_size,
            num_workers=num_workers,
            initializer=init_analyzation_worker  
        )
    
    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Analysis with", num_workers, "worker took", length, "seconds!")
    
    # Save analyzed dataset
    analyzed_dataset.save_to_disk(birdnetlib_analyzed_data_path)

    print("Finished analyzing audio birdnetlib!")

    print("Analysis with", num_workers, "worker took", length, "seconds!")

if __name__ == '__main__':
  main()