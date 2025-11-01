# Python script creating polyphonic dataset from a BirdSet train set
# args: 
# - BirdSet Dataset key (eg. 'HSN') [problem BirdSet necessary, ] or datasetpath?
# - output_dir for dataset

from datasets import load_from_disk, Dataset, Audio, Features, Sequence, Value, concatenate_datasets
import tempfile
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

from source_separation import separate_audio
from utils import validate_species_tag, get_most_confident_detection, get_best_source_idx, get_validated_sources, only_target_bird_detected, extract_relevant_bounds, flatten_features, generate_batches, balance_dataset_by_species, birdset_code_to_ebird_taxonomy
from dsp import analyze_with_birdnetlib, detect_call_bounds, stft_mask_bandpass, segment_audio, remove_segments_without_events, num_samples_to_duration_s, pad_audio_end


def separate_example(example, separation_session_data=None):

    # get audio, filename and ebird-code
    audio = example["audio"] # acces audio_array by audio['array'] and sampling_rate by audio['sampling_rate']

    # do source separation
    session, input_node, output_node = separation_session_data
    sources, source_sr = separate_audio(session,
                   input_node,
                   output_node,
                   audio['array'],
                   input_sampling_rate=audio['sampling_rate']
                   )
    
    for i, src in enumerate(sources):
        example[f"source_{i}"] = src
    example['sources_sampling_rate'] = source_sr

    return example

def extract_clean_sources_from_example(example, birdset_subset):
    
    # get all sources
    i = 0
    sources = []
    while f"source_{i}" in example:
        sources.append(np.array(example[f"source_{i}"]))
        i += 1
    sources_sampling_rate = example['sources_sampling_rate']

    birdset_code = example["ebird_code"]
    
    # analyze audio with birdnetlib
    list_of_detections_per_source = []
    for source_array in sources:
        detections = analyze_with_birdnetlib(source_array, sources_sampling_rate)
        list_of_detections_per_source.append(detections)

    # get all good sources
    validated_sources_data = get_validated_sources(list_of_detections_per_source, example, birdset_subset)

    if not validated_sources_data:
        return None

    call_examples = []

    for validated_source in validated_sources_data:

        source_idx = validated_source['source_index']
        source_array = sources[source_idx]
        # get on-/offsets
        call_bounds = detect_call_bounds(source_array, sr=sources_sampling_rate,
                        smooth_ms=25,
                        threshold_ratio=0.1,
                        min_gap_s=0.02,
                        min_call_s=0.1)

        # get bounding boxes and mask audio
        audio_filtered, time_frequency_bounds = stft_mask_bandpass(source_array, sources_sampling_rate, n_fft=1024, low_pct=2, high_pct=98, events=call_bounds)

        # store filtered audio and time-frequency bounds
        call_examples.append({
            'audio_filtered': {"array": audio_filtered, "sampling_rate": sources_sampling_rate},
            'time_freq_bounds': time_frequency_bounds,
            'scientific_name': validated_source['scientific_name'],
            'original_birdset_subset': birdset_subset,
            'original_file': example['filepath']
            # - common name
            # - timestamps original file - start, end
        })
    
    return call_examples

def extract_segments_from_example(example, birdset_subset):

    filename = Path(example['filepath']).name
    
    # get all sources
    i = 0
    sources = []
    while f"source_{i}" in example:
        sources.append(np.array(example[f"source_{i}"]))
        i += 1
    sources_sampling_rate = example['sources_sampling_rate']

    birdset_code = example["ebird_code"]
    
    # analyze audio with birdnetlib
    list_of_detections_per_source = []
    for source_array in sources:
        detections = analyze_with_birdnetlib(source_array, sources_sampling_rate)
        list_of_detections_per_source.append(detections)

    # get all good sources
    validated_sources_data = get_validated_sources(list_of_detections_per_source, example, birdset_subset)

    if not validated_sources_data:
        return None

    segmented_example = []

    for validated_source in validated_sources_data:

        source_idx = validated_source['source_index']
        source_array = sources[source_idx]

        # Get on-/offsets
        call_bounds = detect_call_bounds(source_array, sr=sources_sampling_rate,
                        smooth_ms=25,
                        threshold_ratio=0.1,
                        min_gap_s=0.02,
                        min_call_s=0.1)
        
        # Skip processing if no call bounds detected
        if not call_bounds:
            continue

        # Get bounding boxes and mask audio
        audio_filtered, time_frequency_bounds = stft_mask_bandpass(source_array, sources_sampling_rate, n_fft=1024, low_pct=2, high_pct=98, events=call_bounds)

        # Segment audio
        segment_length= 5
        audio_segments = segment_audio(audio_filtered, sources_sampling_rate, segment_length, keep_incomplete=True) 

        # Handle last segment
        last_segment = audio_segments[-1]
        segment_duration = num_samples_to_duration_s(len(last_segment['audio_array']), sources_sampling_rate)

        # If the segment is at least 1s, pad the end to reach desired length
        if segment_duration >= 1.0:
            padded_array, pad_end_s = pad_audio_end(last_segment['audio_array'], sources_sampling_rate, segment_length)
            last_segment['audio_array'] = padded_array

        # Otherwise, remove it entirely
        else:
            audio_segments = audio_segments[:-1]

        # Remove segments without events
        segments_with_events = remove_segments_without_events(audio_segments, call_bounds)

        # Remove segments with detections of other birds
        detections = list_of_detections_per_source[source_idx]
        birdset_code = validated_source['birdset_code']
        ebird_code, common_name, scientific_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)

        for segment in segments_with_events:
            start_time = segment['start_time']
            end_time = segment['end_time']
            if only_target_bird_detected(detections, scientific_name, start_time, end_time, confidence_threshold=0.0):
                segment_to_return = {
                    "audio": {'array': segment['audio_array'], 'sampling_rate': sources_sampling_rate},
                    "time_freq_bounds": extract_relevant_bounds(start_time, end_time, time_frequency_bounds),
                    'birdset_code': birdset_code,
                    'ebird_code': ebird_code,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                    "original_birdset_subset": birdset_subset,
                    "original_file": filename
                    }

                segmented_example.append(segment_to_return)   
    
    return segmented_example

def preprocess_example(example, birdset_subset, separation_session_data=None):

    # get audio, filename and ebird-code
    audio = example["audio"] # acces audio_array by audio['array'] and sampling_rate by audio['sampling_rate']
    filename = Path(audio["path"]).name
    birdset_code = example["ebird_code"]

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
    is_validated, birdset_label, birdnet_detection = validate_species_tag(birdset_code, birdset_subset, most_confident_detection)

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

    # # # Load from BirdSet
    # raw_dataset = load_dataset('DBD-research-group/BirdSet', 'HSN_xc', split='train', trust_remote_code=True)
    # raw_dataset = raw_dataset.cast_column("audio", Audio()) # original sampling rate of 32kHz is preserved
    
    # # TODO: remove this once processing whole datase
    # # For now 
    # # ------------------------
    # # Store/Load Raw Dataset/Subset
    # # ------------------------
    
    # subset = raw_dataset.select(range(0, 1))

    # # Save raw dataset
    # print("")
    # output_path = "datasets/raw_subset"
    # subset.save_to_disk(output_path)
    # print(f"Saved raw dataset to {output_path}")
    # print("")

    
    # print("Load raw subset from disk...")
    # subset = load_from_disk('datasets/raw_subset')
    # print(subset)
    # print('')

    # # ------------------------
    # # Separate Audio 
    # # ------------------------

    # Save the current cache path
    # original_cache = datasets.config.HF_DATASETS_CACHE

    # # Load source separation model
    # session, input_node, output_node = load_separation_model(model_dir="resources/bird_mixit_model_checkpoints/output_sources4", 
    #                         checkpoint="resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090")
    # separation_session_data = (session, input_node, output_node)

    # # Store with on-/offset, frequency bounds in original file
    # with tempfile.TemporaryDirectory() as temp_cache_dir:

    #     separate_fn = partial(
    #         separate_example,
    #         separation_session_data=separation_session_data
    #     )

    #     separated_dataset = process_in_batches(
    #                     subset,
    #                     process_fn=separate_fn,
    #                     cache_dir=temp_cache_dir,
    #                 )
        
    # print(separated_dataset)

    # # Save preprocessed dataset
    # output_path = "datasets/separated_subset"
    # separated_dataset.save_to_disk(output_path)
    # print("")
    # print(f"Saved separated dataset to {output_path}")
    # print("")

    # separated_dataset = load_from_disk('datasets/separated_subset')

    # # ------------------------------
    # # Segment sources
    # # ------------------------------

    # segments_dataset_rows = {
    # "audio": [],
    # "time_freq_bounds": [],
    # "birdset_code": [],
    # "ebird_code": [],
    # "scientific_name": [],
    # "common_name": [],
    # "original_birdset_subset": [],
    # "original_file": []
    # }

    # for example in tqdm(separated_dataset):
    #     segments = extract_segments_from_example(example, birdset_subset="HSN")
    #     if segments:
    #         for row in segments:  # each row is a dict with a single audio dict
    #             segments_dataset_rows["audio"].append(row["audio"])
    #             segments_dataset_rows["time_freq_bounds"].append(row["time_freq_bounds"])
    #             segments_dataset_rows["ebird_code"].append(row["ebird_code"])
    #             segments_dataset_rows["birdset_code"].append(row["birdset_code"])
    #             segments_dataset_rows["scientific_name"].append(row["scientific_name"])
    #             segments_dataset_rows["common_name"].append(row["common_name"])
    #             segments_dataset_rows["original_birdset_subset"].append(row["original_birdset_subset"])
    #             segments_dataset_rows["original_file"].append(row["original_file"])

    # segments_dataset = Dataset.from_dict(segments_dataset_rows)
    
    # # --------------------------
    # # Store Preprocessed Dataset
    # # --------------------------
    
    # # Save preprocessed dataset
    # output_path = "datasets/segments_subset"
    # segments_dataset.save_to_disk(output_path)
    # print("")
    # print(f"Saved segments dataset to {output_path}")
    # print("")

    # ------------------------
    # Mix Audio 
    # ------------------------

     # Load raw dataset
    segments_data = load_from_disk('datasets/segments_subset')
    sampling_rate = segments_data[0]['audio']['sampling_rate']
    segments_data = segments_data.cast_column("audio", Audio(sampling_rate=sampling_rate))
    balanced_dataset = balance_dataset_by_species(segments_data)
    print(balanced_dataset)

     # Save preprocessed dataset
    output_path = "datasets/balanced_subset"
    balanced_dataset.save_to_disk(output_path)
    print("")
    print(f"Saved balanced dataset to {output_path}")
    print("")

    # Setup features
    raw_features = balanced_dataset.features
    flattened_raw_features = flatten_features("raw_files", raw_features)

    mix_features = Features({
        "id": Value("string"),
        "audio": Sequence(Value("float32")),
        "sampling_rate": Value("int32"),
        "polyphony_degree": Value("int32"),
        "birdset_code_multilabel": Sequence(Value("int32")),
        **flattened_raw_features
    })

    # Generate batches
    print("Mix audio in batches", flush=True)

    max_polyphony_degree = 6
    segment_length_in_s = 5
    random_seed = 42

    temp_dirs = []
    for i, batch in enumerate(generate_batches(balanced_dataset, 
                                               max_polyphony_degree, segment_length_in_s, 
                                               sampling_rate, random_seed=random_seed)):
        ds = Dataset.from_list(batch, features=mix_features)
        print(f'finished batch {i}')
        
        tmp_dir = tempfile.mkdtemp(prefix=f"mix_batch_{i}_")
        ds.save_to_disk(tmp_dir)
        temp_dirs.append(tmp_dir)

    # Concatenate batches
    print("Concatenate batches", flush=True)
    datasets = [load_from_disk(d) for d in temp_dirs]
    full_dataset = concatenate_datasets(datasets)

    # Save final dataset
    polyphonic_dataset_path = 'datasets/polyphonic_subset'
    full_dataset.save_to_disk(polyphonic_dataset_path)
    print(f'Saved mixed dataset to {polyphonic_dataset_path}', flush=True)

    # Load dataset
    new_dataset = load_from_disk(polyphonic_dataset_path)
    print(f'New Dataset saved to disk: {new_dataset}', flush=True)

    # Remove tmp files
    for d in temp_dirs:
        shutil.rmtree(d)

    # After the block ends, restore it
    # datasets.config.HF_DATASETS_CACHE = original_cache

    # ------------------------
    # Store Polyphonic Dataset
    # ------------------------


if __name__ == '__main__':
  main()