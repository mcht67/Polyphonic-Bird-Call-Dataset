from datasets import load_from_disk, Audio
from omegaconf import OmegaConf

from modules.dsp import encode_audio_dict

print("Start dataset upload...")

# Load the parameters from the config file
cfg = OmegaConf.load("params.yaml")
segmented_data_path = cfg.paths.segmented_data
polyphonic_data_path = cfg.paths.polyphonic_data
subset = cfg.dataset.subset
is_balanced = cfg.mix.balance_dataset
huggingface_user = cfg.upload.huggingface_user
huggingface_dataset_name = cfg.upload.huggingface_dataset_name

huggingface_path = huggingface_user + "/" + huggingface_dataset_name

polyphonic_dataset = load_from_disk(polyphonic_data_path)
dataset_name = subset + "_balanced" if is_balanced else subset
commit_message_polyphonic = f"updates polyphonic dataset with in {dataset_name}"
polyphonic_dataset.push_to_hub(huggingface_path, config_name=dataset_name, private=True, commit_message=commit_message_polyphonic)


# segmented_dataset = load_from_disk(segmented_data_path)
# sampling_rate = segmented_dataset[0]["audio"]["sampling_rate"]

# def encode_example(example):
#     audio = example["audio"]
#     audio_bytes, sampling_rate = encode_audio_dict(audio)
    
#     # Update audio field with encoded bytes
#     example["audio"] = {
#         "bytes": audio_bytes,
#         "path": None,
#         "sampling_rate": sampling_rate
#     }
    
#     return example

# segmented_dataset = segmented_dataset.map(
#     encode_example,
#     num_proc=4,
#     desc="Encoding audio"
# )

# commit_message_segmented = f"updates raw segments dataset in {subset}"
# segmented_dataset.push_to_hub(huggingface_path, config_name=subset+'_segments', private=True, commit_message=commit_message_segmented)

print("Completed dataset upload.")