import numpy as np
import os
import tensorflow.compat.v1 as tf
import numpy as np
from librosa import resample

from integrations.birdnetlib.analyze import get_most_confident_detection, check_dominant_species
from integrations.birdset.utils import validate_species_tag_multi

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
        
        audio_array = resample(audio_array,
                               orig_sr=input_sampling_rate,
                               target_sr=target_sr)

    audio_array = audio_array.astype(np.float32)
    audio_array = audio_array[np.newaxis, np.newaxis, :]  # [1, 1, samples]

    output_numpy = session.run(output_node,
                               feed_dict={input_node: audio_array})

    output_numpy = np.squeeze(output_numpy, axis=0)  # [sources, samples]
    return output_numpy, target_sr

# def separate_batch(session,
#                   input_node,
#                   output_node,
#                   audio_list,
#                   input_sampling_rate):

#     results = []
#     for audio in audio_list:
#         separated, output_sampling_rate = separate_audio(session,
#                                        input_node,
#                                        output_node,
#                                        audio,
#                                        input_sampling_rate)
#         results.append(separated)
#     return results, output_sampling_rate

# def separate_batch_stacked(session,
#                          input_node,
#                          output_node,
#                          audio_list,
#                          sampling_rate,
#                          target_sr=22050):
#     """
#     Process a batch of numpy arrays using a single model execution.

#     Returns:
#         output_batch: numpy array with shape [N, num_sources, samples]
#         target_sr: sampling rate (22.05 kHz for this model)
#     """

#     processed = []

#     # Preprocess individually: enforce mono, resample, convert dtype
#     for audio in audio_list:
#         if audio.ndim != 1:
#             audio = np.mean(audio, axis=0)

#         if sampling_rate != target_sr:
#             audio = resample(audio, orig_sr=sampling_rate, target_sr=target_sr)

#         processed.append(audio.astype(np.float32))

#     # Determine maximum length and pad to uniform size
#     max_len = max(len(x) for x in processed)
#     padded = [np.pad(x, (0, max_len - len(x))) for x in processed]

#     # Stack into a batch tensor [N, 1, samples]
#     batch_input = np.stack(padded, axis=0)
#     batch_input = batch_input[:, np.newaxis, :]

#     # Forward pass once for all inputs
#     output_batch = session.run(output_node, feed_dict={input_node: batch_input})

#     # Shape typically becomes [N, num_sources, samples]
#     return output_batch, target_sr

# def separate_examples_batch(examples, separation_session_data, target_sr=22050):
#     """
#     Process a batch of examples (dicts with 'audio' key) using separate_batch_stacked.
#     Returns updated examples with 'sources' key filled.
#     """
#     session, input_node, output_node = separation_session_data

#     # Collect audio arrays
#     audio_arrays = [audio['array'] for audio in examples['audio']]
#     sampling_rates = [audio['sampling_rate'] for audio in examples['audio']]

#     # If sampling rates differ, resample each audio to target_sr
#     audio_arrays_resampled = []
#     for audio, sr in zip(audio_arrays, sampling_rates):
#         if sr != target_sr:
#             from librosa import resample
#             audio = resample(audio, orig_sr=sr, target_sr=target_sr)
#         audio_arrays_resampled.append(audio)

#     # Run batch separation
#     separated_batch, output_sr = separate_batch_stacked(
#         session, input_node, output_node,
#         audio_arrays_resampled, target_sr, target_sr=target_sr
#     )

#     # Build sources back into examples
#     updated_examples = []
#     for ex, separated in zip(examples, separated_batch):
#         ex['sources'] = [
#             {"audio": {"array": np.array(source), "sampling_rate": output_sr}, "detections": []}
#             for source in separated
#         ]
#         updated_examples.append(ex)

#     return updated_examples


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
    
    example['sources'] = [{"audio": {"array": np.array(source), "sampling_rate": source_sr}, "detections": []} for source in sources]

    return example

# def separate_batch(batch, model_dir, checkpoint):

#     # Load source separation model
#     session, input_node, output_node = load_separation_model(model_dir=model_dir, 
#                             checkpoint=checkpoint)
#     separation_session_data = (session, input_node, output_node)

#     results = []
#     for example in batch:
#         example = separate_example(example, separation_session_data)
#     return batch

# def separate_batch(batch, model_dir, checkpoint):
#     session, input_node, output_node = load_separation_model(model_dir=model_dir, 
#                                                               checkpoint=checkpoint)
#     separation_session_data = (session, input_node, output_node)
    
#     processed_examples = []
#     for example in batch:
#         processed_example = separate_example(example, separation_session_data)
#         processed_examples.append(processed_example)
    
#     return processed_examples 

# Global variable for worker processes
_separation_session = None

# def init_separation_worker(model_dir, checkpoint):
#     """Initialize model once per worker process"""
#     global _separation_session
#     _separation_session = load_separation_model(model_dir=model_dir, 
#                                                  checkpoint=checkpoint)
#     print(f"Worker {os.getpid()} initialized model")

def init_separation_worker(model_dir, checkpoint):
    """Initialize model once per worker process"""
    print(f"DEBUG: init_separation_worker called")
    print(f"  Received model_dir: {model_dir}")
    print(f"  Received checkpoint: {checkpoint}")

    import os
    import tensorflow as tf
    
    print(f"Worker {os.getpid()} starting initialization")
    print(f"  model_dir: {model_dir}")
    print(f"  checkpoint: {checkpoint}")  # Debug: verify it's not None
    
    # Disable TensorFlow parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    
    global _separation_session
    _separation_session = load_separation_model(model_dir, checkpoint=checkpoint)
    print(f"Worker {os.getpid()} initialized model successfully")

def separate_batch(batch):
    """Process batch using pre-loaded model"""
    print("Start separating batch")
    global _separation_session
    
    results = []
    for i, example in enumerate(batch):
        print('process example', i)
        processed_example = separate_example(example, _separation_session)
        results.append(processed_example)
    
    return results

# def separate_batch(batch_dataset, model_dir, checkpoint):
#     """
#     Processes a batch (HuggingFace Dataset) inside a worker process.
#     Loads TF model once per worker, then processes all examples.
#     """
#     # Load TF1 separation model (safe inside worker)
#     session, input_node, output_node = load_separation_model(
#         model_dir=model_dir, 
#         checkpoint=checkpoint
#     )
#     separation_session_data = (session, input_node, output_node)

#     # Convert HF Dataset → python list of dict examples
#     batch = batch_dataset.to_list()

#     processed_examples = []

#     for example in batch:
#         processed = separate_example(example, separation_session_data)
#         processed_examples.append(processed)

#     return processed_examples


# def separate_examples_parallel(examples, separation_session_data):
#     """
#     Process a list of examples in parallel using Python multiprocessing.
#     Each example is processed individually with the TF1 model.
#     """

#     from multiprocessing import Pool, cpu_count

#     # Worker function
#     def worker(example):
#         return separate_example(example, separation_session_data)

#     # Number of processes
#     n_processes = min(cpu_count(), len(examples))

#     with Pool(processes=n_processes) as pool:
#         results = pool.map(worker, examples)

#     return results


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

