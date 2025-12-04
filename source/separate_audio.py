from datasets import load_from_disk, Audio, Features, Sequence, Value
from omegaconf import OmegaConf
from functools import partial
import shutil
import time

from integrations.bird_mixit.source_separation import separate_batch, init_separation_worker
from modules.dataset import process_batches_in_parallel


def main():

    print("Start separating audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    raw_data_path = cfg.paths.raw_data
    source_separated_data_path = cfg.paths.source_separated_data

    model_dir = cfg.source_separation.model_dir
    checkpoint = cfg.source_separation.checkpoint
    # model_dir = os.path.abspath(model_dir)
    # checkpoint = os.path.abspath("resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path)
    raw_dataset = raw_dataset.select(range(80))
    raw_dataset = raw_dataset.cast_column("audio", Audio()) # original sampling rate of 32kHz is preserved

    # # Load source separation model
    # session, input_node, output_node = load_separation_model(model_dir=model_dir, 
    #                         checkpoint=checkpoint)
    # separation_session_data = (session, input_node, output_node)

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

    # Define cache dir
    temp_cache_dir = source_separated_data_path + '_cache'

    # separate_fn = partial(
    #     separate_example,
    #     separation_session_data=separation_session_data
    # )

    # NUM_PROC = 1
    # separated_dataset = process_in_batches(
    #                 raw_dataset,
    #                 process_fn=separate_fn,
    #                 cache_dir=temp_cache_dir,
    #                 num_proc=NUM_PROC
                # )

    # separate_batch_fn = partial(separate_batch, model_dir=model_dir, checkpoint=checkpoint)
    
    # separated_dataset = process_batches_in_parallel(raw_dataset, process_batch_fn=separate_batch_fn, batch_size=10)

    # init_fn = partial(init_separation_worker, model_dir=model_dir, checkpoint=checkpoint)

    # separated_dataset = process_batches_in_parallel(
    #     raw_dataset,
    #     process_batch_fn=separate_batch,
    #     initializer=init_fn,
    #     initargs=(model_dir, checkpoint),
    #     num_workers=1
    # )

    for num_workers in [8,4,2,1]:

        batch_size = int(len(raw_dataset) / num_workers)
        # Calculate the start time
        start = time.time()

        # NEW - pass separate_batch directly, no partial needed
        separated_dataset = process_batches_in_parallel(
            raw_dataset, 
            process_batch_fn=separate_batch,  # No partial!
            batch_size=batch_size,
            num_workers=num_workers,
            initializer=init_separation_worker,
            initargs=(model_dir, checkpoint)  # Pass params here
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

        # try:
        #     shutil.rmtree(temp_cache_dir)
        # except Exception as e:
        #     print(f"Warning: cache cleanup failed: {e}")


if __name__ == '__main__':
#   import multiprocessing
#   multiprocessing.set_start_method("spawn", force=True)
  main()