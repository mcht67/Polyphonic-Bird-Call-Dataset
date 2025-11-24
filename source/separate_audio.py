import tempfile
from datasets import load_from_disk, Audio
from omegaconf import OmegaConf
from functools import partial

from integrations.bird_mixit.runner import load_separation_model, separate_example
from modules.dataset import process_in_batches

def main():
    # Load the parameters from the config file
    cfg = OmegaConf.load("config.yaml")
    dataset_name = cfg.dataset.name
    base_folder = cfg.dataset.base_folder
    raw_folder = cfg.dataset.raw_folder
    separated_folder = cfg.dataset.separated_folder

    # Load raw dataset
    dataset_path = base_folder + dataset_name + '/'
    raw_dataset = load_from_disk(dataset_path + raw_folder)
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
                        cache_dir=temp_cache_dir,
                    )

    # Save separated dataset
    separated_dataset.save_to_disk(base_folder + separated_folder)


if __name__ == '__main__':
  main()