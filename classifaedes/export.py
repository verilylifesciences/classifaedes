# Copyright 2019 Verily Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build the inference graph, load a checkpoint, and export."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from classifaedes import hparams_lib
from classifaedes import inputs_lib
from classifaedes import metadata
from classifaedes import model_lib
import classifaedes.simple_image_lib as sil
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import learn as contrib_learn

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('export_dir_base', None,
                    '(optional) Export destination; defaults to '
                    '[model_dir]/export_/.')
flags.DEFINE_string('model_dir', None, 'Training output_dir.')


def make_input_fn(image_shape):
  """Returns an Estimator input_fn for the exported serving graph."""

  def _prep_input(png):
    with tf.name_scope('prep_input'):
      img = tf.image.decode_png(png, channels=1)
      return inputs_lib.prep_image(img, image_shape)

  def input_fn():
    """Estimator input_fn for serving.

    Returns:
      InputFnOps - a namedtuple of features, labels, and default_inputs.
    """
    pngs = tf.placeholder(tf.string, shape=[None])
    imgs = tf.map_fn(_prep_input, pngs, dtype=tf.float32)

    inputs = {
        'encoded_pngs': pngs,
        'images': imgs,   # For model_fn.
    }
    return contrib_learn.utils.input_fn_utils.InputFnOps(
        inputs, None, {'inputs': pngs})
  return input_fn


def run_test_inference(image_shape, export_dir):
  """Run a dummy prediction using the exported model under export_dir."""
  with tf.Session(graph=tf.Graph()) as sess:
    meta_graph_def = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    sig = meta_graph_def.signature_def[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    inputs_name = sig.inputs[
        tf.saved_model.signature_constants.PREDICT_INPUTS].name
    outputs_name = sig.outputs[
        tf.saved_model.signature_constants.PREDICT_OUTPUTS].name

    dummy_image = sil.encode_png(np.ones(image_shape, dtype='uint8'))
    return sess.run(outputs_name, {inputs_name: [dummy_image]})


def run_export(hps, image_shape, model_dir, export_dir_base):
  """Export model checkpoint under `model_dir` to `export_dir_base`."""
  estimator = contrib_learn.Estimator(
      model_fn=model_lib.build_model_fn(hps, {}),
      model_dir=model_dir,
  )
  export_dir = estimator.export_savedmodel(
      export_dir_base=export_dir_base,
      serving_input_fn=make_input_fn(image_shape),
  )
  tf.logging.info('Exported SavedModel to %s', export_dir)
  print('Exported SavedModel to', export_dir)
  return export_dir


def main(unused_argv):
  hps = hparams_lib.load_from_file(FLAGS.model_dir)
  input_md = metadata.load_metadata(model_dir=FLAGS.model_dir)
  image_shape = metadata.shape_from_metadata(input_md)

  export_dir_base = FLAGS.export_dir_base or (FLAGS.model_dir + '/export_')
  export_dir = run_export(hps, image_shape, FLAGS.model_dir, export_dir_base)
  # Run in-process inference on a dummy image to sanity check our export.
  dummy_prediction = run_test_inference(image_shape, export_dir)
  tf.logging.info('Dummy prediction: %s', dummy_prediction)


if __name__ == '__main__':
  tf.app.run()
