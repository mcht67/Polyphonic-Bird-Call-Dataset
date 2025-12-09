from datasets import load_from_disk, Audio, Features, Sequence, Value
from omegaconf import OmegaConf
import psutil
import time
from math import ceil

from integrations.bird_mixit.source_separation import separate_batch, init_separation_worker
from modules.dataset import process_batches_in_parallel, overwrite_dataset
from modules.utils import monitor_memory

def add_sources_column(dataset):
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
        empty_sources = [None] * len(dataset)
        return dataset.add_column("sources", empty_sources, feature=source_features["sources"])


def main():

    print("Start preparing dataset for separation...")

    # # Start memory monitoring thread
    # stop_event = threading.Event()
    # monitor_thread = threading.Thread(target=monitor_memory, args=(5, stop_event))
    # monitor_thread.daemon = True
    # monitor_thread.start()

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    raw_data_path = cfg.paths.raw_data
    #source_separated_data_path = cfg.paths.source_separated_data
    source_separation_meta_data_path = cfg.paths.source_separation_metadata

    model_dir = cfg.source_separation.model_dir
    checkpoint = cfg.source_separation.checkpoint
    sampling_rate = cfg.source_separation.sampling_rate
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path)
    raw_dataset = raw_dataset.select(range(100))
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Truncate audio arrays to 30s max
    # Get the sampling rate from the Audio feature
    MAX_DURATION = 30
    sr = raw_dataset.features["audio"].sampling_rate
    max_len = int(sr * MAX_DURATION)

    def truncate_example(example):
        arr = example["audio"]["array"]
        example["audio"]["array"] = arr[:max_len]  # drop anything beyond 30s
        return example

    print("Truncate examples.")
    raw_dataset = raw_dataset.map(truncate_example)

    # Check if column sources exists, else add column
    if  not "sources" in raw_dataset.column_names:
        raw_dataset = add_sources_column(raw_dataset)
        
    # Get processing configuration
    def get_num_workers(gb_per_worker=1, cpu_percentage=0.7):
        num_workers_cpu_max = psutil.cpu_count(logical=False) or psutil.cpu_count()
        num_workers_cpu = cpu_percentage * num_workers_cpu_max

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        num_workers_memory = int(available_gb/gb_per_worker)
        return min(num_workers_cpu, num_workers_memory) 
    
    num_workers = get_num_workers(gb_per_worker=1)
    batch_size = ceil((len(raw_dataset) + 1) / num_workers)

    print("Prepared dataset for Separation.")

    # Calculate the start time
    print("Start separating audio...")
    start = time.time()

    # Separate sources
    separated_dataset = process_batches_in_parallel(
        raw_dataset, 
        process_batch_fn=separate_batch,
        batch_size=batch_size,
        num_workers=num_workers,
        initializer=init_separation_worker,
        initargs=(model_dir, checkpoint)
    )

    # # Stop monitoring memory
    # stop_event.set()
    # monitor_thread.join(timeout=1)

    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Separation with", num_workers, "worker took", length, "seconds!")

    # Save separated dataset
    overwrite_dataset(separated_dataset, raw_data_path, store_backup=False)
    # print(f"Attempt saving dataset to {source_separated_data_path}.")
    # separated_dataset.save_to_disk(source_separated_data_path)
    # print(f"Succesfully saved dataset to {source_separated_data_path}.")


    # Store stage results for dvc tracking
    sources_subset = separated_dataset.select_columns(["sources"])
    sources_subset.to_json(source_separation_meta_data_path)

    print("Finished separating audio!")

if __name__ == '__main__':
  main()