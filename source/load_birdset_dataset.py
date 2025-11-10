from datasets import load_dataset, Dataset
import argparse

# Collect arguments
parser = argparse.ArgumentParser(description="Load BirdSet dataset and store local copies.")
parser.add_argument("--birdset_subset", type=str, required=True,
                    help="BirdSet subset key (eg. 'HSN', 'HSN_xc',...)"),
parser.add_argument("--split", type=str, required=True,
                    help="BirdSet split (eg. 'train', 'test', 'test_5s'...)")
parser.add_argument("--output_path", type=str, required=True,
                    help="Output path to store dataset at.")
args = parser.parse_args()
                    
# Load dataset
dataset = load_dataset('DBD-research-group/BirdSet', args.birdset_subset, split=args.split, trust_remote_code=True)

# Store dataset on disk
dataset.save_to_disk(args.output_path)