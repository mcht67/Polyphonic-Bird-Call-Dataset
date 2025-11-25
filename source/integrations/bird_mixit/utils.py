from source.integrations.birdnet.utils import get_most_confident_detection, check_dominant_species
from source.integrations.birdset.utils import validate_species_tag_multi
import numpy as np

def get_best_source_idx(list_of_detections_per_source, birdset_example=None, birdset_subset=None, decision_rule=None):
    """
    Takes list of detections per source and an optional decision rule. Chooses best source based in decision rule chosen.
    Per default 'highest_confidence_single_detection' is chosen as decision rule, choosing the source with the detection with highest confidence over all detectins of all sources.

    Parameters:
        list of detections per source : list of dicts
        decision rule: str

    Returns:
        index of chosen source
    """

    if decision_rule == None:
        decision_rule = 'highest_confidence_single_detection'
    
    if decision_rule == 'highest_confidence_single_detection':
        most_confident_detections = [None for i in range(len(list_of_detections_per_source))]

        for idx, detections in enumerate(list_of_detections_per_source):
            if detections:
                most_confident_detections[idx] = get_most_confident_detection(detections)

                # highest_confidence_idx = np.argmax([detection['confidence'] for detection in detections])
                # most_confident_detection[idx] = detections[highest_confidence_idx]

        best_source_idx = np.argmax([detection['confidence'] if detection is not None else -np.inf 
                                     for detection in most_confident_detections ])
        
    # if decision_rule == 'confidence_threshold_species_percentile':
        
    #     for idx, detections in enumerate(list_of_detections_per_source):
    #         if detections:
    #             # Get detections with confidence above threshold → removes uncertain detections
    #             detections = [detection for detection in detections if (detection['confidence'] > 0.9)]

    #             # Choose source when 0.1 - 0.9 percentile of detections above threshold are one species
    #             species_tags = [detection['scientific_name'] for detection in detections]
    #             scientific_name, is_dominant = check_dominant_species(detections)

    #             # check if species is in BirdSet labels
    #             validate_species_tag(birdset_code, birdset_subset, scientific_name=scientific_name)

    #              # extract all events (start, end) where detection is above threshold
    #              # and return it for later comparison with detected call bounds (compare time and species)
                
    # TODO:
    # introduce treshold
    #
    # add other decision rules:
    # - chossing source with the highest mean confidence over 5-10 highest confidence detections
    # - choosing source with highest summed confidence over 5-10 highest confidence detections
    # - choosing source with only detections of one bird
    #
    # Only use detections of the bird we are searching for?
    #
    # Use all sources with high confidence scores for one specific bird? As long as it is tagged in birdset??

    return best_source_idx 

def get_validated_sources(list_of_detections_per_source, birdset_example, birdset_subset, confidence_threshold=0.9, min_detection_percentage=0.9):

    sources = []
    for source_idx, detections in enumerate(list_of_detections_per_source):
        if detections:
            # Get detections with confidence above threshold → removes uncertain detections
            confident_detections = [detection for detection in detections if (detection['confidence'] > confidence_threshold)]

            # Choose source when 90% of detections are above threshold and refer to the same species
            detected_species = [detection['scientific_name'] for detection in confident_detections]
            dominant_species, is_dominant = check_dominant_species(detected_species, min_detection_percentage)

            if is_dominant:

                # check if species is in BirdSet labels
                birdset_species_ids = [birdset_example['ebird_code']] + birdset_example['ebird_code_multilabel']
                is_validated, birdset_code, comparison_label = validate_species_tag_multi(birdset_species_ids, birdset_subset, scientific_name=dominant_species)

                if is_validated:
                    # extract all events (start, end) where detection is above threshold
                    detection_bounds = [(detection['start_time'], detection['end_time']) for detection in confident_detections if detection['scientific_name']==dominant_species]

                    # Check if there is a source with this species already
                    same_species_indices = [idx for idx, source in enumerate(sources) if source['scientific_name'] == dominant_species]
                    if len(same_species_indices) > 1:
                        raise ValueError("Expected not more than one match")
                   
                    if not same_species_indices:
                        sources.append({'source_index': source_idx, 'birdset_code': birdset_code, 'scientific_name': dominant_species, 'detection_bounds': detection_bounds})
                    else: 
                        # Replace source if the new one has more detections
                        same_species_idx = same_species_indices[0]
                        source_with_same_species = sources[same_species_idx]
                        if len(detection_bounds) > len(source_with_same_species['detection_bounds']):
                            sources[same_species_idx] = source_with_same_species

    return sources