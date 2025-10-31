# Python script creating polyphonic dataset from a BirdSet train set
# args: 
# - BirdSet Dataset key (eg. 'HSN') [problem BirdSet necessary, ] or datasetpath?
# - output_dir for dataset

from datasets import Audio, load_dataset, load_from_disk
import datasets
from functools import partial
import tempfile
from pathlib import Path
import numpy as np

from source_separation import load_separation_model, separate_audio
from utils import process_in_batches, validate_species_tag, get_most_confident_detection, get_best_source_idx
from dsp import analyze_with_birdnetlib, detect_call_bounds, stft_mask_bandpass, plot_save_mel_spectrogram


def preprocess_example(example, birdset_subset, separation_session_data=None):

    # get audio, filename and ebird-code
    audio = example["audio"] # acces audio_array by audio['array'] and sampling_rate by audio['sampling_rate']
    filename = Path(audio["path"]).name
    ebird_code = example["ebird_code"]

    # do source separation
    session, input_node, output_node = separation_session_data
    sources, source_sr = separate_audio(session,
                   input_node,
                   output_node,
                   audio['array'],
                   input_sampling_rate=audio['sampling_rate']
                   )

    # analyze audio with birdnetlib
    list_of_detections_per_source = []
    for source_array in sources:
        detections = analyze_with_birdnetlib(source_array, source_sr)
        list_of_detections_per_source.append(detections)

    # choose source
    best_source_idx = get_best_source_idx(list_of_detections_per_source)
    best_source = sources[best_source_idx]

    # validate species
    most_confident_detection = get_most_confident_detection(list_of_detections_per_source[best_source_idx])
    is_validated, birdset_label, birdnet_detection = validate_species_tag(ebird_code, birdset_subset, most_confident_detection)

    # TODO: 
    # - all calls should be validated, not just highest confidence ones 
    # - this should also be part of choosing the source -> sources with only one bird should be higher ranked

    if is_validated:
        # get on-/offsets
        call_bounds = detect_call_bounds(best_source, sr=source_sr,
                        smooth_ms=25,
                        threshold_ratio=0.1,
                        min_gap_s=0.02,
                        min_call_s=0.1)

        # get bounding boxes and mask audio
        audio_filtered, time_frequency_bounds = stft_mask_bandpass(best_source, source_sr, n_fft=1024, low_pct=2, high_pct=98, events=call_bounds)

        # store filtered audio and time-frequency bounds
        example['audio_filtered'] = audio_filtered
        example['time_freq_bounds'] = time_frequency_bounds
        #plot_save_mel_spectrogram(audio_filtered, source_sr, f"{filename} [{birdset_label}]")
    else:
        example['audio_filtered'] = None
        example['time_freq_bounds'] = None
        print(f'Species could not be validated. Birdset-Label: {birdset_label}, BirdNet-Detection: {birdnet_detection}')

    return example

def main():
    # ------------------------
    # Load Raw Dataset 
    # ------------------------

    # TODO: [BirdSet env necessary! or load from disk?? and store under specific path?] 
    # -> load in subprocess with specific BirdSet environment if necessary
    # -> store on disk

    # # Load from BirdSet
    raw_dataset = load_dataset('DBD-research-group/BirdSet', 'HSN_xc', split='train', trust_remote_code=True)
    raw_dataset = raw_dataset.cast_column("audio", Audio()) # original sampling rate of 32kHz is preserved
    
    # TODO: remove this once processing whole datase
    # For now 
    # ------------------------
    # Store/Load Raw Dataset/Subset
    # ------------------------
    
    subset = raw_dataset.select(range(0, 1))

    # Save raw dataset
    print("")
    output_path = "datasets/raw_subset"
    subset.save_to_disk(output_path)
    print(f"Saved raw dataset to {output_path}")
    print("")

    # print("")
    # print("Load raw subset from disk...")
    # subset = load_from_disk('datasets/raw_subset')
    # print(subset)
    # print('')

    # ------------------------
    # Preprocess Audio 
    # ------------------------

    # Save the current cache path
    original_cache = datasets.config.HF_DATASETS_CACHE

    # Load source separation model
    session, input_node, output_node = load_separation_model(model_dir="resources/bird_mixit_model_checkpoints/output_sources4", 
                            checkpoint="resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
    separation_session_data = (session, input_node, output_node)

    # Store with on-/offset, frequency bounds in original file
    with tempfile.TemporaryDirectory() as temp_cache_dir:

        preprocess_fn = partial(
            preprocess_example,
            birdset_subset = 'HSN', 
            separation_session_data=separation_session_data
        )

        modified_dataset = process_in_batches(
                        subset,
                        process_fn=preprocess_fn,
                        cache_dir=temp_cache_dir,
                    )

    # ------------------------
    # Store Preprocessed Dataset
    # ------------------------

    print(modified_dataset
          )
    # Save preprocessed dataset
    output_path = "datasets/preprocessed_subset"
    modified_dataset.save_to_disk(output_path)
    print("")
    print(f"Saved preprocessed dataset to {output_path}")
    print("")

    new_dataset = load_from_disk('datasets/preprocessed_subset')
    print(new_dataset)

    # After the block ends, restore it
    datasets.config.HF_DATASETS_CACHE = original_cache

    # ------------------------
    # Mix Audio 
    # ------------------------

    # ------------------------
    # Store Polyphonic Dataset
    # ------------------------


if __name__ == '__main__':
  main()