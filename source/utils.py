from datasets import concatenate_datasets
from pathlib import Path
import json
import os
import pandas as pd

def process_in_batches(dataset, process_fn, cache_dir, prefix="", batch_size=100):
    """
    Generic batch processor for datasets.
    
    Args:
        dataset: Dataset to process.
        process_fn: Function to apply to each batch.
        cache_dir: Directory to store batch caches.
        prefix: Optional prefix for cache filenames.
        batch_size: Number of samples per batch.
    
    Returns:
        Concatenated processed dataset.
    """
    processed_datasets = []
    total_samples = len(dataset)
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}...")
        
        batch_dataset = dataset.select(range(i, end_idx))
        cache_file = os.path.join(cache_dir, f"{prefix}_batch_{i}_{end_idx}_cache.arrow")
        batch_processed = batch_dataset.map(process_fn, cache_file_name=cache_file)
        
        processed_datasets.append(batch_processed)
    
    print(f"Concatenating {len(processed_datasets)} batches...")
    return concatenate_datasets(processed_datasets)

def validate_species_tag(birdset_id, birdset_subset, birdnetlib_detection):
    """
    Takes the birdset label which is an id, the birdset subset used and a birdnetlib detection. 
    It gives back True or False depending on birdset label and birdnetlib detection are the same species.

    Parameters:
        birdset_id: int
        birdset_subset: str
        birdnetlibn_dection: dict

    Returns:
        True or False based on species tags matching, birdset label, birdnet detection
    """
    ebird_code, common_name, sci_name = birdset_id_to_ebird_taxonomy(birdset_id, birdset_subset)
    birdset_label = common_name + ', ' + sci_name

    if birdnetlib_detection==None:
        return False, birdset_label, None

    predicted_common_name = normalize_name(birdnetlib_detection['common_name'])
    predicted_sci_name = normalize_name(birdnetlib_detection['scientific_name'])

    if predicted_common_name == common_name and predicted_sci_name == sci_name:
        validated = True
    else:
        validated = False

    birdnet_detection = predicted_common_name + ', ' + predicted_sci_name

    return validated, birdset_label, birdnet_detection

def normalize_name(name):
    return name.strip().lower().replace(" ", "_")

def get_audio_file_name(output_dir, filename, details):
    return f"{output_dir}{Path(filename).stem}_{details}.wav"

def birdset_id_to_ebird_taxonomy(birdset_id, dataset_key):
    # Load BirdSet label mapping
    with open(f"resources/birdset_ebird_codes/{dataset_key}_ebird_codes.json") as f:
        birdset_labels = json.load(f)

    ebird_code = birdset_labels['id2label'][str(birdset_id)]

    # Load the official eBird taxonomy file
    tax = pd.read_csv("resources/ebird_taxonomy_v2024.csv")

    match = tax[tax["SPECIES_CODE"].str.lower() == ebird_code.lower()]
    if not match.empty:
        ebird_code = match.iloc[0]["SPECIES_CODE"].lower()  # e.g. 'rebunt1'
        common_name = normalize_name(match.iloc[0]["PRIMARY_COM_NAME"])
        sci_name = normalize_name(match.iloc[0]["SCI_NAME"])

    return ebird_code, common_name, sci_name

def get_most_confident_detection(detections):
    """
    Gets most confident detection for a list of detections provided by birdnetlib.

    Parameters:
        detections: list of dicts

    Returns:
        most confident detection: dict
    """
    if detections is None or len(detections) == 0:
        return None
    
    valid_indices = [i for i, d in enumerate(detections) if d is not None]
    if valid_indices:
        most_confident_detection_idx = max(valid_indices, key=lambda i: detections[i]['confidence'])
        return detections[most_confident_detection_idx]
    else:
        return None
    
    