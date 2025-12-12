import numpy as np
import os
import tensorflow.compat.v1 as tf
import numpy as np
from librosa import resample
import gc

from modules.utils import log_memory
# from integrations.birdnetlib.analyze import get_most_confident_detection, check_dominant_species
# from integrations.birdset.utils import validate_species_tag_multi

"""
Functions to run source separation using a pretrained MixIT or similar model.

This version refactors the original Google Research `process_wav.py` script
so that it can be called programmatically, not just via CLI.

Example:

    from bird_mixit import process_wav
    from datasets import load_dataset

    # Load an example HuggingFace Audio object
    ds = load_dataset("ashraq/esc50")  # for example
    audio_obj = ds["train"][0]["audio"]

    model = process_wav.load_model("/path/to/model_dir")

    separated = process_wav.separate_sources(audio_obj, model)
"""


def load_separation_model(model_dir,
                          input_tensor='input_audio/receiver_audio:0',
                          output_tensor='denoised_waveforms:0',
                          checkpoint=None):

    tf.disable_v2_behavior()
    tf.reset_default_graph()

    meta_graph_filename = os.path.join(model_dir, 'inference.meta')
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph(meta_graph_filename)
        meta_graph_def = graph.as_graph_def()

    checkpoint = checkpoint or tf.train.latest_checkpoint(model_dir)

    session = tf.Session(graph=graph)
    saver.restore(session, checkpoint)

    input_node = graph.get_tensor_by_name(input_tensor)
    output_node = graph.get_tensor_by_name(output_tensor)

    return session, input_node, output_node


def separate_audio(session,
                   input_node,
                   output_node,
                   audio_array,
                   input_sampling_rate,
                   target_sr=22050):

    # Enforce mono
    if audio_array.ndim != 1:
        audio_array = np.mean(audio_array, axis=0)

    # Resample if needed
    if input_sampling_rate != target_sr:
        print("Need to resample from", input_sampling_rate, "kHz to", target_sr, "kHz. This is inefficient. Resampling should be done beforehand.")
        audio_array = resample(audio_array,
                               orig_sr=input_sampling_rate,
                               target_sr=target_sr)

    audio_array = audio_array.astype(np.float32)
    audio_array = audio_array[np.newaxis, np.newaxis, :]  # [1, 1, samples]

    output_numpy = session.run(output_node,
                               feed_dict={input_node: audio_array})

    output_numpy = np.squeeze(output_numpy, axis=0)  # [sources, samples]
    return output_numpy, target_sr

def separate_example(example, separation_session_data=None):

    # get audio, filename and ebird-code
    audio = example["audio"] # acces audio_array by audio['array'] and sampling_rate by audio['sampling_rate']

    # do source separation
    session, input_node, output_node = separation_session_data
    sources, source_sr = separate_audio(session,
                   input_node,
                   output_node,
                   audio['array'],
                   input_sampling_rate=int(audio['sampling_rate'])
                   )
    
     # Convert sources efficiently without extra copies
    example['sources'] = [
        {
            "audio": {
                "array": source.copy().astype(np.float32),
                "sampling_rate": source_sr
            }#, 
            #"detections": [] #[{"common_name": None, "scientific_name": None, "label": None, "confidence": None, "start_time": None, "end_time": None}]
        } 
        for source in sources
    ]
    
    # Explicitly delete the original sources array to free memory
    del sources
    
    #example['sources'] = [{"audio": {"array": np.array(source), "sampling_rate": source_sr}, "detections": []} for source in sources]

    return example

# Global variable for worker processes
_separation_session = None

def init_separation_worker(model_dir, checkpoint):
    """Initialize model once per worker process"""

    import os
    import tensorflow as tf
    
    print(f"Worker {os.getpid()} starting initialization")
    print(f"  model_dir: {model_dir}")
    print(f"  checkpoint: {checkpoint}")
    
    # Disable TensorFlow parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    
    global _separation_session
    _separation_session = load_separation_model(model_dir, checkpoint=checkpoint)
    print(f"Worker {os.getpid()} initialized model successfully")

def separate_batch(batch):
    """Process batch using pre-loaded model"""
    print("Worker", os.getpid(), ": Start separating batch")
    global _separation_session
    
    results = []
    batch_size = len(batch)
    for i, example in enumerate(batch):
        print("Worker", os.getpid(), ": Separating example", i+1, "of ", batch_size)
        processed_example = separate_example(example, _separation_session)
        results.append(processed_example)

        # Clean up after each example
        del processed_example

        print("Separated example:")
        log_memory()
    
    # Force garbage collection after batch
    gc.collect()
    
    print("Worker", os.getpid(), ": Finished separating batch")
    
    return results

############################
# Legacy
############################

# def complete_separation_numpy(input,
#                         sampling_rate,
#                         #output,
#                         model_dir,
#                         #input_channels=0,
#                         num_sources=2,
#                         output_channels=0,
#                         input_tensor='input_audio/receiver_audio:0', 
#                         output_tensor='denoised_waveforms:0',
#                         #scale_input=False,
#                         checkpoint=None,
#                         #write_outputs_separately=True,
#                         ):
#     """
#     Run source separation. Mimics runnging the script with args.

#     Args:
#     input: numpy audio_array (mono)
        
#     Returns:
#     output_numpy: list of numpy arrays with all sources
#     sampling_rate: sampling_rate
        
#     """
#     tf.disable_v2_behavior()

#     meta_graph_filename = os.path.join(model_dir, 'inference.meta')
#     tf.logging.info('Importing meta graph: %s', meta_graph_filename)

#     with tf.Graph().as_default() as g:
#         saver = tf.train.import_meta_graph(meta_graph_filename)
#         meta_graph_def = g.as_graph_def()

#     tf.reset_default_graph()
#     input_wav = np.array(input, dtype=np.float32)

#     # Enforce mono 
#     if not input_wav.ndim == 1:
#         input_wav = np.mean(input_wav, axis=0)

#     # Resample to 22.05kz as expected from the source separation model
#     expected_sampling_rate = 22050
#     if not sampling_rate == expected_sampling_rate:
#         print("Resample to 22.05kz as expected from the source separation model")
#         input_wav = resample(input_wav, orig_sr=sampling_rate, target_sr=expected_sampling_rate)

#     input_wav = tf.expand_dims(input_wav, axis=0)  
#     input_wav = tf.expand_dims(input_wav, axis=0) # shape: [1, 1, samples]
#     output_wav, = tf.import_graph_def(
#         meta_graph_def,
#         name='',
#         input_map={input_tensor: input_wav}, # args.input_tensor: default='input_audio/receiver_audio:0'
#         return_elements=[output_tensor]) # args.ouput_tenspr: default='denoised_waveforms:0'

#     # Remove writing section, keep computation
#     output_wav = tf.squeeze(output_wav, 0)  # [sources, samples]

#     checkpoint = checkpoint or tf.train.latest_checkpoint(model_dir)

#     with tf.Session() as sess:
#         tf.logging.info('Restoring from checkpoint: %s', checkpoint)
#         saver.restore(sess, checkpoint)
#         output_numpy = sess.run(output_wav)

#     return output_numpy, expected_sampling_rate

# def get_best_source_idx(list_of_detections_per_source, birdset_example=None, birdset_subset=None, decision_rule=None):
#     """
#     Takes list of detections per source and an optional decision rule. Chooses best source based in decision rule chosen.
#     Per default 'highest_confidence_single_detection' is chosen as decision rule, choosing the source with the detection with highest confidence over all detectins of all sources.

#     Parameters:
#         list of detections per source : list of dicts
#         decision rule: str

#     Returns:
#         index of chosen source
#     """

#     if decision_rule == None:
#         decision_rule = 'highest_confidence_single_detection'
    
#     if decision_rule == 'highest_confidence_single_detection':
#         most_confident_detections = [None for i in range(len(list_of_detections_per_source))]

#         for idx, detections in enumerate(list_of_detections_per_source):
#             if detections:
#                 most_confident_detections[idx] = get_most_confident_detection(detections)

#                 # highest_confidence_idx = np.argmax([detection['confidence'] for detection in detections])
#                 # most_confident_detection[idx] = detections[highest_confidence_idx]

#         best_source_idx = np.argmax([detection['confidence'] if detection is not None else -np.inf 
#                                      for detection in most_confident_detections ])
        
#     # if decision_rule == 'confidence_threshold_species_percentile':
        
#     #     for idx, detections in enumerate(list_of_detections_per_source):
#     #         if detections:
#     #             # Get detections with confidence above threshold → removes uncertain detections
#     #             detections = [detection for detection in detections if (detection['confidence'] > 0.9)]

#     #             # Choose source when 0.1 - 0.9 percentile of detections above threshold are one species
#     #             species_tags = [detection['scientific_name'] for detection in detections]
#     #             scientific_name, is_dominant = check_dominant_species(detections)

#     #             # check if species is in BirdSet labels
#     #             validate_species_tag(birdset_code, birdset_subset, scientific_name=scientific_name)

#     #              # extract all events (start, end) where detection is above threshold
#     #              # and return it for later comparison with detected call bounds (compare time and species)
                
#     # TODO:
#     # introduce treshold
#     #
#     # add other decision rules:
#     # - chossing source with the highest mean confidence over 5-10 highest confidence detections
#     # - choosing source with highest summed confidence over 5-10 highest confidence detections
#     # - choosing source with only detections of one bird
#     #
#     # Only use detections of the bird we are searching for?
#     #
#     # Use all sources with high confidence scores for one specific bird? As long as it is tagged in birdset??

#     return best_source_idx 

# def get_validated_sources(list_of_detections_per_source, birdset_example, birdset_subset, confidence_threshold=0.9, min_detection_percentage=0.9):

#     sources = []
#     for source_idx, detections in enumerate(list_of_detections_per_source):
#         if detections:
#             # Get detections with confidence above threshold → removes uncertain detections
#             confident_detections = [detection for detection in detections if (detection['confidence'] > confidence_threshold)]

#             # Choose source when 90% of detections are above threshold and refer to the same species
#             detected_species = [detection['scientific_name'] for detection in confident_detections]
#             dominant_species, is_dominant = check_dominant_species(detected_species, min_detection_percentage)

#             if is_dominant:

#                 # check if species is in BirdSet labels
#                 birdset_species_ids = [birdset_example['ebird_code']] + birdset_example['ebird_code_multilabel']
#                 is_validated, birdset_code, comparison_label = validate_species_tag_multi(birdset_species_ids, birdset_subset, scientific_name=dominant_species)

#                 if is_validated:
#                     # extract all events (start, end) where detection is above threshold
#                     detection_bounds = [(detection['start_time'], detection['end_time']) for detection in confident_detections if detection['scientific_name']==dominant_species]

#                     # Check if there is a source with this species already
#                     same_species_indices = [idx for idx, source in enumerate(sources) if source['scientific_name'] == dominant_species]
#                     if len(same_species_indices) > 1:
#                         raise ValueError("Expected not more than one match")
                   
#                     if not same_species_indices:
#                         sources.append({'source_index': source_idx, 'birdset_code': birdset_code, 'scientific_name': dominant_species, 'detection_bounds': detection_bounds})
#                     else: 
#                         # Replace source if the new one has more detections
#                         same_species_idx = same_species_indices[0]
#                         source_with_same_species = sources[same_species_idx]
#                         if len(detection_bounds) > len(source_with_same_species['detection_bounds']):
#                             sources[same_species_idx] = source_with_same_species

#     return sources

