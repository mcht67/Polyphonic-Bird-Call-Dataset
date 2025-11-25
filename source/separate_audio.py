import tempfile
from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
from functools import partial

from integrations.bird_mixit.runner import load_separation_model, separate_example
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

    # Store with on-/offset, frequency bounds in original file
    with tempfile.TemporaryDirectory() as temp_cache_dir:

        separate_fn = partial(
            separate_example,
            separation_session_data=separation_session_data
        )

        separated_dataset = process_in_batches(
                        raw_dataset,
                        process_fn=separate_fn,
                        cache_dir=temp_cache_dir
                    )

    # Save separated dataset
    separated_dataset.save_to_disk(source_separated_data_path)

    print("Finished separating audio!")

if __name__ == '__main__':
  main()