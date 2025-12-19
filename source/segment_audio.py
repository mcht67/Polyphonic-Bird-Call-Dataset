import numpy as np
from tqdm import tqdm
from datasets import Dataset, load_from_disk, load_dataset, Sequence, Value, Features
from pathlib import Path
from omegaconf import OmegaConf
import time
import os
import shutil
import gc

from modules.dsp import detect_event_bounds, stft_mask_bandpass, segment_audio, pad_audio_end, num_samples_to_duration_s, remove_segments_without_events, extract_relevant_bounds
from modules.utils import get_num_workers
from modules.dataset import process_batches_in_parallel, move_dataset

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
        # audio = sources['audio'][source_idx]
        # source_array = audio['array']
        # source_sampling_rate = audio['sampling_rate']

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

def extract_segments_from_batch(batch):
    print("Worker", os.getpid(), "Start segmenting batch")
    
    all_segments = []
    for example in batch:
        segments = extract_segments_from_example(example, _segment_length_in_s, _birdset_subset)
        if segments:
            all_segments.append(segments)

        # Delete segments
        del segments

    # Force garbage collection
    gc.collect()

    print("Worker", os.getpid(), "Finished segmenting batch")

    return all_segments

def init_segment_worker(segment_length_in_s, birdset_subset):
    print("Start initilization of worker.")
    global _segment_length_in_s
    global _birdset_subset
    _segment_length_in_s = segment_length_in_s
    _birdset_subset = birdset_subset
    print("Initilization of worker succesful.")

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
    #raw_dataset = load_from_disk(raw_data_path) # This should work as long as dataset_info.json and state.json exist
    raw_dataset = load_dataset(
        "arrow",
        data_files=os.path.join(raw_data_path, "data-*.arrow"),
        #streaming=True,
        split="train",
        cache_dir="hf_cache"
    )
    #raw_dataset = raw_dataset['train']

    # Check if column 'sources' and nested feature 'detections' exist in raw_dataset
    if  not "sources" in raw_dataset.column_names:
        raise Exception("Can not segment Dataset. Dataset does not contain column 'sources'.")
    #elif not "detections" in raw_dataset.features['sources'].feature.keys():
    elif not "detections" in raw_dataset.features['sources'][0].keys():
        raise Exception("Can not segment Dataset. Nested feature 'detections' does not exist in column 'sources'.")

    # # Define features of segments dataset
    # segments_dataset_rows = {
    #     "audio": [],
    #     "time_freq_bounds": [],
    #     "birdset_code": [],
    #     "ebird_code": [],
    #     "scientific_name": [],
    #     "common_name": [],
    #     "original_birdset_subset": [],
    #     "original_file": []
    #     }
    
    # Calculate the start time
    start = time.time()

    # def process_shard(shard_id, num_shards):
    # """Each process gets its own shard"""
    # ds = load_dataset(
    #     "arrow",
    #     data_files=os.path.join('../data/HSN/raw', "data-*.arrow"),
    #     split="train",
    #     streaming=True
    # )
    
    # # Each worker processes only its shard
    # ds_shard = ds.shard(num_shards=num_shards, index=shard_id)
    
    # results = []
    # for example in ds_shard:
    #     result = your_processing_function(example)
    #     results.append(result)
    
    # return results

    # # Run across multiple processes
    # num_processes = 4
    # with Pool(num_processes) as pool:
    #     all_results = pool.starmap(
    #         process_shard,
    #         [(i, num_processes) for i in range(num_processes)]
    #     )

    # # Extract segments from shard
    # def extract_segments_from_shard(shard, segment_length_in_s, birdset_subset):
    #     segments = []
    #     for example in tqdm(shard):
    #         if segments:
    #             segments.append(extract_segments_from_example(example, segment_length_in_s, birdset_subset))
    #     return segments

    # Define features of segments dataset

    features = Features({"audio": {
                                    "array": Sequence(Value("float32")),
                                    "sampling_rate": Value("int64"),
                                },
                        "time_freq_bounds": Sequence(Sequence(Value("float32"))),
                        "birdset_code": Value("int64"),
                        "ebird_code": Value("string"),
                        "scientific_name": Value("string"),
                        "common_name": Value("string"),
                        "original_birdset_subset": Value("string"),
                        "original_file": Value("string")
                        })

    # Get multiprocessing configuration
    num_workers = get_num_workers(gb_per_worker=1, cpu_percentage=0.8)
    batch_size = 200 # ceil(((len(raw_dataset) + 1) / num_workers)//10)
    batches_per_shard = 1
    num_batches = (len(raw_dataset) + batch_size - 1) // batch_size
    print("Process", num_batches, "batches with a batch size of", batch_size,
          "on", num_workers, "workers.")


    arrow_dir = process_batches_in_parallel(
        raw_dataset,
        process_batch_fn=extract_segments_from_batch,
        features=features,
        batch_size=batch_size,
        num_batches=num_batches,
        num_workers=num_workers,
        temp_dir="tmp_segment",
        batches_per_shard=batches_per_shard,
        initializer=init_segment_worker,
        initargs=(segment_length_in_s, birdset_subset)
    )

    # # Extract segments from examples
    # for example in tqdm(raw_dataset):
    #     segments = extract_segments_from_example(example, segment_length_in_s, birdset_subset=birdset_subset)
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



    # Calculate the end time and time taken
    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("Segmentation with", num_workers, "workers and", num_batches, "batches with a batch size of", batch_size,
            "took", length, "seconds!")

    # Create segments dataset
    #segments_dataset = Dataset.from_dict(segments_dataset_rows)

    # Store segments dataset
    #segments_dataset.save_to_disk(segmented_data_path)
    #move_dataset(arrow_dir, segmented_data_path, store_backup=False)

    #shutil.rmtree("hf_cache")

    print("Finished segementing audio!")
    
if __name__ == '__main__':
    main()