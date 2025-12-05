import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Working directory: {os.getcwd()}")
print(f"Environment: {os.environ.get('VIRTUAL_ENV', 'No venv')}")


from datasets import load_from_disk, Audio, Features, Sequence, Value
from omegaconf import OmegaConf
import psutil
import time
from math import ceil

from integrations.bird_mixit.source_separation import separate_batch, init_separation_worker
from modules.dataset import process_batches_in_parallel


def main():

    print("Start separating audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    raw_data_path = cfg.paths.raw_data
    source_separated_data_path = cfg.paths.source_separated_data

    model_dir = cfg.source_separation.model_dir
    checkpoint = cfg.source_separation.checkpoint
    sampling_rate = cfg.source_separation.sampling_rate
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path)
    raw_dataset = raw_dataset.select(range(10))
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Define the schema for the new column holding the sources data
    source_features = Features({
        "sources": Sequence({
            "audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int32")}, #Audio(),
            "detections": Sequence({
                "common_name": Value("string"),
                "scientific_name": Value("string"),
                "label": Value("string"),
                "confidence": Value("float32"),
                "start_time": Value("float32"),
                "end_time": Value("float32"),
            })
        })
    })

    # Update dataset schema
    empty_sources = [None] * len(raw_dataset)
    raw_dataset = raw_dataset.add_column("sources", empty_sources, feature=source_features["sources"])

    num_workers = psutil.cpu_count(logical=False) or psutil.cpu_count()
    batch_size = ceil((len(raw_dataset) + 1) / num_workers)

    # Calculate the start time
    start = time.time()

    # NEW - pass separate_batch directly, no partial needed
    separated_dataset = process_batches_in_parallel(
        raw_dataset, 
        process_batch_fn=separate_batch,
        batch_size=batch_size,
        num_workers=num_workers,
        initializer=init_separation_worker,
        initargs=(model_dir, checkpoint)
    )

    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Separation with", num_workers, "worker took", length, "seconds!")

    # Save separated dataset
    print(f"Attempt saving dataset to {source_separated_data_path}.")
    separated_dataset.save_to_disk(source_separated_data_path)
    print(f"Succesfully saved dataset to {source_separated_data_path}.")

    print("Finished separating audio!")

if __name__ == '__main__':
  main()