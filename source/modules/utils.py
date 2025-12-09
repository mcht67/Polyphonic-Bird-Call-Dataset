from pathlib import Path
import random
from functools import wraps
import psutil
import os
import time

def log_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def monitor_memory(interval=5, stop_event=None):
    """Monitor memory usage of main process and all child processes"""
    main_process = psutil.Process(os.getpid())
    
    while not (stop_event and stop_event.is_set()):
        try:
            # Get main process memory
            main_mem = main_process.memory_info().rss / 1024 / 1024  # MB
            
            # Get all child processes memory
            children = main_process.children(recursive=True)
            children_mem = sum(child.memory_info().rss / 1024 / 1024 for child in children)
            
            # Total system memory usage
            system_mem = psutil.virtual_memory()
            
            print(f"\n{'='*60}")
            print(f"Main process:     {main_mem:>10.2f} MB")
            print(f"Child processes:  {children_mem:>10.2f} MB ({len(children)} workers)")
            print(f"Total (program):  {main_mem + children_mem:>10.2f} MB")
            print(f"System total:     {system_mem.used / 1024 / 1024:>10.2f} MB / {system_mem.total / 1024 / 1024:.2f} MB")
            print(f"System available: {system_mem.available / 1024 / 1024:>10.2f} MB ({system_mem.percent}% used)")
            print(f"{'='*60}\n")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        time.sleep(interval)

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
        :param indices: List of indices.
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