import json
import pandas as pd

from modules.utils import normalize_name

def validate_species_tag(birdset_code, birdset_subset, birdnetlib_detection=None, scientific_name=None, common_name=None):
    """
    Takes the birdset label which is an id, the birdset subset used and a birdnetlib detection or scientific name/common name. 
    It gives back True or False depending on birdset label and birdnetlib detection or given names refer to the same species.
    If Birdnetlib detection is given, it is given priority over common name and scientific name.

    Parameters:
        birdset_code: int
        birdset_subset: str
        birdnetlibn_dection: dict

    Returns:
        True or False based on species tags matching, birdset label, birdnet detection
    """
    ebird_code, birdset_common_name, birdset_sci_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)
    birdset_label = birdset_common_name + ', ' + birdset_sci_name

    if birdnetlib_detection:
        common_name = normalize_name(birdnetlib_detection['common_name'])
        scientific_name = normalize_name(birdnetlib_detection['scientific_name'])

    if common_name == birdset_common_name and scientific_name == birdset_sci_name:
        validated = True
    else:
        validated = False

    comparision_label = common_name + ', ' + scientific_name

    return validated, birdset_label, comparision_label

def validate_species_tag_multi(birdset_codes, birdset_subset, birdnetlib_detection=None, scientific_name=None, common_name=None):
    """
    Takes in birdset codes, the birdset subset used and a birdnetlib detection or scientific name/common name. 
    It gives back True or False depending on one of the birdset codes and the birdnetlib detection or given names refering to the same species.
    If Birdnetlib detection is given, it is given priority over common name and scientific name.

    Parameters:
        birdset_codes: list of int
        birdset_subset: str
        birdnetlibn_dection: dict

    Returns:
        True or False based on species tags matching, birdset label, comparison label
    """
    if common_name: common_name = normalize_name(common_name)
    if scientific_name: scientific_name = normalize_name(scientific_name)

    for birdset_code in birdset_codes:
        ebird_code, birdset_common_name, birdset_sci_name = birdset_code_to_ebird_taxonomy(birdset_code, birdset_subset)
        birdset_common_name = normalize_name(birdset_common_name)
        birdset_sci_name = normalize_name(birdset_sci_name)
        # birdset_label = birdset_common_name + ', ' + birdset_sci_name

        if birdnetlib_detection:
            common_name = normalize_name(birdnetlib_detection['common_name'])
            scientific_name = normalize_name(birdnetlib_detection['scientific_name'])

        is_validated_through_both_names = common_name == birdset_common_name and scientific_name == birdset_sci_name
        is_validated_through_sci_name = False
        is_validated_through_common_name = False

        if not common_name:
            common_name = 'common name not given'
            is_validated_through_sci_name = scientific_name == birdset_sci_name
        
        if not scientific_name:
            scientific_name = 'scientific name not given'
            is_validated_through_common_name = common_name == birdset_common_name

        if  is_validated_through_both_names or is_validated_through_sci_name or is_validated_through_common_name:
            validated = True
        else:
            validated = False

        comparison_label = common_name + ', ' + scientific_name

        return validated, birdset_code, comparison_label

def birdset_code_to_ebird_taxonomy(birdset_code, dataset_key):
    # Load BirdSet label mapping
    with open(f"resources/birdset_ebird_codes/{dataset_key}_ebird_codes.json") as f:
        birdset_labels = json.load(f)

    ebird_code = birdset_labels['id2label'][str(birdset_code)]

    # Load the official eBird taxonomy file
    tax = pd.read_csv("resources/ebird_taxonomy_v2024.csv")

    match = tax[tax["SPECIES_CODE"].str.lower() == ebird_code.lower()]
    if not match.empty:
        ebird_code = match.iloc[0]["SPECIES_CODE"].lower()  # e.g. 'rebunt1'
        common_name = normalize_name(match.iloc[0]["PRIMARY_COM_NAME"])
        sci_name = normalize_name(match.iloc[0]["SCI_NAME"])

    return ebird_code, common_name, sci_name