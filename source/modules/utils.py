from pathlib import Path
import random
from functools import wraps

def with_random_state(func):
    """
    Decorator that allows a function to accept random_state parameter.
    The function can accept either a seed (int) or a state tuple.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract random_state from kwargs
        random_state = kwargs.pop('random_state', None)
        
        if random_state is None:
            # No state provided, call function normally
            return func(*args, **kwargs)
        
        # Save current state
        current_state = random.getstate()
        
        try:
            # Set the provided state
            if isinstance(random_state, int):
                # It's a seed
                random.seed(random_state)
            else:
                # It's a state tuple
                random.setstate(random_state)
            
            # Call the original function
            return func(*args, **kwargs)
        
        finally:
            # Restore original state
            random.setstate(current_state)
    
    return wrapper

def normalize_name(name):
    return name.strip().lower().replace(" ", "_")

def get_audio_file_name(output_dir, filename, details):
    return f"{output_dir}{Path(filename).stem}_{details}.wav"
class IndexMap:
    """Random non-repeating index generator with optional auto-reset."""

    def __init__(self, indices, random_seed=None, auto_reset=False):
        """
        :param indices: List of indices
        :param random_seed: Seed for reproducible sequence.
        :param auto_reset: If True, reshuffles when exhausted.
        """
        self.auto_reset = auto_reset
        self.rng = random.Random(random_seed)
        self.indices = indices
        self.num_indices = len(self.indices)
        self.pos = 0
        self.reset()

    def reset(self):
        """Reshuffle indices and restart sequence."""
        self.rng.shuffle(self.indices)
        self.pos = 0

    def pop_random(self):
        """
        Return next random index.
        Auto-resets if enabled when exhausted.
        """
        if self.pos >= self.num_indices:
            if self.auto_reset:
                self.reset()
            else:
                raise IndexError("No unused indices left")

        value = self.indices[self.pos]
        self.pos += 1
        return value

@with_random_state
def create_index_map(num_indices, random_state=None):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param num_indices: Number of indices of the dict to create
            :type num_indices: int
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''

    # Handle random state if provided
    if random_state is not None:
        current_state = None
        if isinstance(random_state, int):
            # It's a seed
            current_state = random.getstate()
            random.seed(random_state)
        else:
            # It's a state tuple
            current_state = random.getstate()
            random.setstate(random_state)
    try:
        # Setup index map
        indices = list(range(num_indices))
        random.shuffle(indices)
        index_map = dict(zip(indices, [False] * len(indices)))
        return index_map
    finally:
        # Restore original state if changed
        if random_state is not None and current_state is not None:
            random.setstate(current_state)

@with_random_state
def create_index_map_from_range(range):
    '''
            Creates dictionary of integers and booleans corresponding to used and unused indices initialized to False. 
    
            :param range: Range of indices of the dict to create
            :type range: range
            
            :return: Dictionary with indices as keys and booleans as values
            :rtype: dict of int: bool
    '''

    # Setup random indexing
    indices = list(range)
    random.shuffle(indices)
    index_map = dict(zip(indices, [False] * len(indices)))
    return index_map

def pop_random_index(index_map):
    '''
            Gets next false key from index map and sets it to True. 
            
            :return: Pseudo random index
            :rtype: int
    '''
    # Find the keys where the value is False
    false_keys = [key for key, value in index_map.items() if value is False]

    first_key = false_keys[0]

    index_map[first_key] = True

    return first_key

def reset_index_map(index_map):
    '''
            Resets the index map by setting all values to False
    '''
    for key, value in index_map.items():
        index_map[key] = False

