import subprocess
import tensorflow.compat.v1 as tf

# Load and save model

tf.disable_v2_behavior()

ckpt_path = "/resources/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090"
meta_graph_path = "/resources/bird_mixit_model_checkpoints/output_sources4/inference.meta"
export_dir = "/resources/bird_mixit_savedmodel"

from pathlib import Path
import os
# script_dir = Path(os.getcwd())
# print(script_dir)
# Go two levels up
filepath = os.path.realpath(__file__)
base_dir = str(Path(filepath).parents[3])
print(base_dir)

meta_graph_path = base_dir  + meta_graph_path
ckpt_path = base_dir + ckpt_path
export_dir = base_dir + export_dir
#print(meta_graph_path)

# filepath = os.path.realpath(__file__)
# print(filepath)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(sess, ckpt_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.SERVING],
        strip_default_attrs=True
    )
    builder.save()

cmd = [
    "conda", "run", "-n", "onnx_model_conversion",
    "python",
    "-m", "tf2onnx.convert",
    "--saved-model", "resources/bird_mixit_savedmodel",
    "--output", "resources/bird_mixit_onnx/model.onnx",
    "--opset", "13"
]

subprocess.run(cmd, check=True)


# cmd = [
#             "conda", "run", "-n", "onnx_model_conversion",
#             "python", 
#             "-m onnxruntime.tools.optimizer_cli",
#             "--input", "resources/bird_mixit_onnx/model_optimized.onnx",
#             "--output", "resources/bird_mixit_onnx/model_optimized.onnx"
#         ]

# subprocess.run(cmd)


