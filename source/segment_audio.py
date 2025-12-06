import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from pathlib import Path
from omegaconf import OmegaConf

from modules.dsp import detect_event_bounds, stft_mask_bandpass, segment_audio, pad_audio_end, num_samples_to_duration_s, remove_segments_without_events, extract_relevant_bounds, plot_save_mel_spectrogram

from integrations.birdnetlib.detections import check_dominant_species, only_target_bird_detected
from integrations.birdset.utils import validate_species_tag_multi, birdset_code_to_ebird_taxonomy

def get_validated_sources(example, birdset_subset, confidence_threshold=0.2, min_detection_percentage=0.1):

    # Init validated sources
    validated_sources = []

    for source_idx, source in enumerate(example['sources']):
        detections = source['detections']
        if detections:
            # Get detections with confidence above threshold → removes uncertain detections
            confident_detections = [detection for detection in detections if (detection['confidence'] > confidence_threshold)]

            # Choose source when 90% of detections are above threshold and refer to the same species
            detected_species = [detection['scientific_name'] for detection in confident_detections]
            dominant_species, is_dominant = check_dominant_species(detected_species, min_detection_percentage)

            if is_dominant:

                # check if species is in BirdSet labels
                birdset_species_ids = [example['ebird_code']] + example['ebird_code_multilabel']
                is_validated, birdset_code, comparison_label = validate_species_tag_multi(birdset_species_ids, birdset_subset, scientific_name=dominant_species)

                if is_validated:
                    # extract all events (start, end) where detection is above threshold
                    detection_bounds = [(detection['start_time'], detection['end_time']) for detection in confident_detections if detection['scientific_name']==dominant_species]

                    # Check if there is a source with this species already
                    same_species_indices = [idx for idx, source in enumerate(validated_sources) if source['scientific_name'] == dominant_species]
                    if len(same_species_indices) > 1:
                        raise ValueError("Expected not more than one match")
                   
                    if not same_species_indices:
                        validated_sources.append({'source_index': source_idx, 'birdset_code': birdset_code, 'scientific_name': dominant_species, 'detection_bounds': detection_bounds})
                    else: 
                        # Replace source if the new one has more detections
                        same_species_idx = same_species_indices[0]
                        source_with_same_species = validated_sources[same_species_idx]
                        if len(detection_bounds) > len(source_with_same_species['detection_bounds']):
                            validated_sources[same_species_idx] = source_with_same_species

    return validated_sources

def extract_segments_from_example(example, segment_length_in_s, birdset_subset):

    filename = Path(example['filepath']).name

    # get all sources
    sources = example['sources']

    # get all good sources
    validated_sources_data = get_validated_sources(example, birdset_subset)

    if not validated_sources_data:
        return None

    segmented_example = []

    for validated_source in validated_sources_data:

        source_idx = validated_source['source_index']
        source = sources[source_idx]
        source_array = np.array(source['audio']['array'])
        source_sampling_rate = source['audio']['sampling_rate']

        # Get on-/offsets
        call_bounds = detect_event_bounds(source_array, sr=source_sampling_rate,
                        smooth_ms=25,
                        threshold_ratio=0.1,
                        min_gap_s=0.02,
                        min_call_s=0.1)
        
        # Skip processing if no call bounds detected
        if not call_bounds:
            continue

        # Get bounding boxes and mask audio
        audio_filtered, time_frequency_bounds = stft_mask_bandpass(source_array, source_sampling_rate, n_fft=1024, low_pct=2, high_pct=98, events=call_bounds)

        # Segment audio
        audio_segments = segment_audio(audio_filtered, source_sampling_rate, segment_length_in_s, keep_incomplete=True) 

        # Handle last segment
        last_segment = audio_segments[-1]
        segment_duration = num_samples_to_duration_s(len(last_segment['audio_array']), source_sampling_rate)

        # If the segment is at least 1s, pad the end to reach desired length
        if segment_duration >= 1.0:
            padded_array, pad_end_s = pad_audio_end(last_segment['audio_array'], source_sampling_rate, segment_length_in_s)
            last_segment['audio_array'] = padded_array

        # Otherwise, remove it entirely
        else:
            audio_segments = audio_segments[:-1]

        # Remove segments without events
        segments_with_events = remove_segments_without_events(audio_segments, call_bounds)

        # Remove segments with detections of other birds
        detections = source['detections']
        birdset_code = validated_source['birdset_code']
        ebird_code, common_name, scientific_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)

        for segment in segments_with_events:
            start_time = segment['start_time']
            end_time = segment['end_time']
            if only_target_bird_detected(detections, scientific_name, start_time, end_time, confidence_threshold=0.0):
                segment_to_return = {
                    "audio": {'array': np.array(segment['audio_array']) , 'sampling_rate': source_sampling_rate},
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

def main():

    print("Start segmenting audio...")

    # Load the parameters from the config file
    cfg = OmegaConf.load("params.yaml")
    # birdnetlib_analyzed_data_path = cfg.paths.birdnetlib_analyzed_data
    raw_data_path = cfg.paths.raw_data
    segmented_data_path = cfg.paths.segmented_data

    birdset_subset = cfg.dataset.subset
    segment_length_in_s = cfg.segmentation.segment_length_in_s

    # Load source separated dataset
    raw_dataset = load_from_disk(raw_data_path)

    # Check if column 'sources' and nested feature 'detections' exist in raw_dataset
    if  not "sources" in raw_dataset.column_names:
        raise Exception("Can not segment Dataset. Dataset does not contain column 'sources'.")
    elif not "detections" in raw_dataset.features['sources'][0].keys():
        raise Exception("Can not segment Datasetx. Nested feature 'detections' does not exist in column 'sources'.")

        raw_dataset = add_sources_column(raw_dataset)

    # Define features of segments dataset
    segments_dataset_rows = {
        "audio": [],
        "time_freq_bounds": [],
        "birdset_code": [],
        "ebird_code": [],
        "scientific_name": [],
        "common_name": [],
        "original_birdset_subset": [],
        "original_file": []
        }

    # Extract segments from examples
    for example in tqdm(raw_dataset):
        segments = extract_segments_from_example(example, segment_length_in_s, birdset_subset=birdset_subset)
        if segments:
            for row in segments:  # each row is a dict with a single audio dict
                segments_dataset_rows["audio"].append(row["audio"])
                segments_dataset_rows["time_freq_bounds"].append(row["time_freq_bounds"])
                segments_dataset_rows["ebird_code"].append(row["ebird_code"])
                segments_dataset_rows["birdset_code"].append(row["birdset_code"])
                segments_dataset_rows["scientific_name"].append(row["scientific_name"])
                segments_dataset_rows["common_name"].append(row["common_name"])
                segments_dataset_rows["original_birdset_subset"].append(row["original_birdset_subset"])
                segments_dataset_rows["original_file"].append(row["original_file"])

    # Create segments dataset
    segments_dataset = Dataset.from_dict(segments_dataset_rows)

    # Store segments dataset
    segments_dataset.save_to_disk(segmented_data_path)

    print("Finished segementing audio!")
    
if __name__ == '__main__':
    main()