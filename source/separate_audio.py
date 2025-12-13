from datasets import load_from_disk, Audio, Features, Sequence, Value
from omegaconf import OmegaConf
import time
from math import ceil
from datetime import datetime

from integrations.bird_mixit.source_separation import separate_batch, init_separation_worker
from modules.dataset import process_batches_in_parallel, overwrite_dataset
from modules.utils import get_num_workers
from modules.dataset import truncate_batch

def add_sources_column(dataset):
   # Define the schema for the new column holding the sources data
        source_features = Features({
            "sources": Sequence({
                "audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int32")}#, #Audio(),
                # "detections": Sequence({
                #     "common_name": Value("string"),
                #     "scientific_name": Value("string"),
                #     "label": Value("string"),
                #     "confidence": Value("float32"),
                #     "start_time": Value("float32"),
                #     "end_time": Value("float32"),
                #})
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
    #source_separated_data_path = cfg.paths.source_separated_data
    source_separation_metadata_path = cfg.paths.source_separation_metadata

    model_dir = cfg.source_separation.model_dir
    checkpoint = cfg.source_separation.checkpoint
    sampling_rate = cfg.source_separation.sampling_rate

    max_duration_s = cfg.source_separation.max_duration_s
    cpu_percentage = cfg.source_separation.cpu_percentage
    gb_per_worker = cfg.source_separation.gb_per_worker
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path)
    #raw_dataset = raw_dataset.select(range(4))
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Remove known bad indices
    bad_indices = [1757]  # Add more if you find them

    raw_dataset = raw_dataset.select([i for i in range(len(raw_dataset)) if i not in bad_indices])

    print(f"Removed {len(bad_indices)} corrupted files")
    print(f"Dataset size: {len(raw_dataset)}")

    # import soundfile as sf
    # import os

    # def check_file(example, idx):
    #     """Find files with corrupted metadata"""
    #     try:
    #         path = example["audio"].get("path")
    #         if path and os.path.exists(path):
    #             # Check what soundfile thinks about this file
    #             info = sf.info(path)
                
    #             # Calculate expected array size
    #             expected_bytes = info.frames * info.channels * 8  # float64
    #             expected_gb = expected_bytes / (1024**3)
                
    #             if expected_gb > 2:  # Anything over 2GB is suspicious
    #                 print(f"\n⚠️  SUSPICIOUS FILE at index {idx}:")
    #                 print(f"   Path: {path}")
    #                 print(f"   Reported duration: {info.duration} seconds")
    #                 print(f"   Reported frames: {info.frames:,}")
    #                 print(f"   Expected array size: {expected_gb:.2f} GB")
    #                 print(f"   Actual file size: {os.path.getsize(path) / (1024**2):.2f} MB")
    #                 return {"suspicious": True}
    #     except Exception as e:
    #         print(f"\n❌ ERROR at index {idx}: {e}")
    #         return {"error": str(e)}
        
    #     return {"ok": True}

    # # Check files around where it crashes (example ~3321-3371)
    # print("Checking all files...")
    # for i in range(1758):
    #     try:
    #         check_file(raw_dataset[i], i)
    #     except Exception as e:
    #         print(f"Can't check index {i}: {e}")
    #         print("Example:", raw_dataset[i])



    #  # Truncate audio arrays to 30s max
    # # Get the sampling rate from the Audio feature
    print("Start truncating examples....")
    num_workers = get_num_workers(cpu_percentage=0.8, gb_per_worker=1.5)
    #truncate_fn = partial(truncate_example, max_duration_s=max_duration_s)
    # raw_dataset.map(truncate_fn, keep_in_memory=False)
    #raw_dataset = raw_dataset.map(truncate_fn, num_proc=num_workers) #, keep_in_memory=False, batched=True, batch_size=100)
    raw_dataset = raw_dataset.map(
        lambda batch: truncate_batch(batch, max_duration_s=max_duration_s),
        batched=True,
        batch_size=30,  # Adjust based on your memory
        writer_batch_size=30,
        num_proc=num_workers,
        keep_in_memory=False
    )

    print("Done")

    # Remove sources column if it exists to remove all old sources and detections
    if  "sources" in raw_dataset.column_names:
        raw_dataset.remove_columns('sources')
    #raw_dataset = add_sources_column(raw_dataset)  
        
    # Get processing configuration  
    num_workers = get_num_workers(gb_per_worker=2, cpu_percentage=0.8)
    batch_size = ceil(((len(raw_dataset) + 1) / num_workers)/10)

    print("Prepared dataset for Separation.")

    # Calculate the start time
    print("Start separating audio...")
    start = time.time()

    features = raw_dataset.features.copy()
    # features['sources'] = Sequence({"audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int64")}})
    features['sources'] = [{"audio": {'array': Sequence(Value("float32")), 'sampling_rate': Value("int64")}}]

    print(features['sources'])

    #print(features['ebird_code_secondary'])
    #print(features['sources'])
    #print(features)
    #raw_dataset.cast_column("ebird_code_multilabel", Sequence(Value("int32")))
    #raw_dataset.cast_column('ebird_code_secondary', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
    #print(raw_dataset.features)
    #raw_dataset.cast(features)

    # Separate sources
    separated_dataset = process_batches_in_parallel(
        raw_dataset,
        process_batch_fn=separate_batch,
        features=features,
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
    overwrite_dataset(separated_dataset, raw_data_path, store_backup=False)

    # # Store stage results for dvc tracking
    sources_subset = separated_dataset.select_columns(["sources"])
    # sources_subset.to_json(source_separation_meta_data_path)

    # Convert to Python objects
    data = sources_subset.to_dict()

    # Add a global timestamp (ISO 8601)
    data["datetime"] = datetime.now().isoformat()

    # Write JSON with timestamp
    import json
    with open(source_separation_metadata_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Finished separating audio!")

if __name__ == '__main__':
  main()