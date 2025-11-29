from datasets import load_from_disk, Audio, Features, Sequence, Value
from omegaconf import OmegaConf
from functools import partial
import shutil

from integrations.bird_mixit.source_separation import load_separation_model, separate_example
from modules.dataset import process_in_batches


def main():

    print("Start separating audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    raw_data_path = cfg.paths.raw_data
    source_separated_data_path = cfg.paths.source_separated_data
    
    # Load raw dataset
    raw_dataset = load_from_disk(raw_data_path)
    raw_dataset = raw_dataset.cast_column("audio", Audio()) # original sampling rate of 32kHz is preserved

    # Load source separation model
    session, input_node, output_node = load_separation_model(model_dir="resources/bird_mixit_model_checkpoints/output_sources4", 
                            checkpoint="resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
    separation_session_data = (session, input_node, output_node)

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

    separate_fn = partial(
        separate_example,
        separation_session_data=separation_session_data
    )

    separated_dataset = process_in_batches(
                    raw_dataset,
                    process_fn=separate_fn,
                    #features=source_features,
                    cache_dir=temp_cache_dir
                )

    # Save separated dataset
    print(f"Attempt saving dataset to {source_separated_data_path}.")
    separated_dataset.save_to_disk(source_separated_data_path)
    print(f"Succesfully saved dataset to {source_separated_data_path}.")

    print("Finished separating audio!")

    try:
        shutil.rmtree(temp_cache_dir)
    except Exception as e:
        print(f"Warning: cache cleanup failed: {e}")


if __name__ == '__main__':
  main()