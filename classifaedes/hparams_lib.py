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
"""Functions for initializing, loading, and saving HParams."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import time


import tensorflow as tf
from tensorflow.contrib import training as contrib_training

flags = tf.app.flags
FLAGS = flags.FLAGS


def defaults():
  """Returns default HParams instance."""
  hps = contrib_training.HParams(
      batch_size=16,

      # Learning rate params.
      lr_init=0.01,
      lr_decay_steps=200,

      # Supported architectures:
      # * inception_v3a: InceptionV3 with corresponding arg-scope.
      # * inception_v3b: InceptionV3 without corresponding arg-scope (notably
      #     excludes batch-norm).
      arch='inception_v3a',
      clip_gradient_norm=0,  # Disabled if zero.
      adam_epsilon=0.1,
      p_dropout=0.1,
      filter_depth_mul=0.3,
      use_global_objective_recall_at_precision=False,
      target_precision=0.9997,
  )
  return hps


def write_to_file(hps, model_dir):
  """Writes HParams instance values to a JSON file under model_dir.

  Format is inverted by hps.parse_json().

  Args:
    hps: HParams.
    model_dir: Model directory.
  """

  hps_path = _hps_path(model_dir)
  tf.logging.info('Recording HParams to path %s.', hps_path)
  with tf.gfile.Open(hps_path, 'w') as fp:
    fp.write(_human_serialize(hps))


def load_from_file(model_dir):
  """Load HParams from `model_dir`.

  Args:
    model_dir: Model directory

  Returns:
    tf.HParams loaded from a JSON file under `model_dir`.
  """
  hps_path = _hps_path(model_dir)
  while not tf.gfile.Exists(hps_path):
    tf.logging.info('Waiting for HParams file to exist at %s.', hps_path)
    time.sleep(10)
  tf.logging.info('Loading HParams from path %s...', hps_path)
  hps = defaults()
  with tf.gfile.Open(hps_path) as fp:
    hps.parse_json(fp.read())

  tf.logging.info('HParams values: \n%s', _human_serialize(hps))
  return hps

#
# Private functions.
#


def _hps_path(model_dir):
  return os.path.join(model_dir, 'hparams.json')


def _human_serialize(hps):
  return json.dumps(hps.values(), indent=2)
