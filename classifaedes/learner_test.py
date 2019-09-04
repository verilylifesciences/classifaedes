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
"""Tests for Debug male/female classification model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile



from absl.testing import flagsaver
from classifaedes import export
from classifaedes import hparams_lib
from classifaedes import learner
from classifaedes import metadata
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class LearnerTest(tf.test.TestCase):

  def testRunExperiment(self):
    """An end-to-end test using tf.learn.Experiment's built-in test."""
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        
        'testdata/data_dir')

    model_dir = tempfile.mkdtemp()
    export_dir_base = tempfile.mkdtemp()

    hps = hparams_lib.defaults()
    hps.batch_size = 2
    hps.filter_depth_mul = 0.1
    hps.arch = 'inception_v3b'
    input_md = metadata.load_metadata(data_dir=data_dir)
    image_shape = metadata.shape_from_metadata(input_md)

    with flagsaver.flagsaver(data_dir=data_dir,
                             shuffle_q_capacity=4,
                             shuffle_q_min_after_deq=2,
                             read_q_capacity=2,
                             train_steps=2,
                             evals_per_ckpt=5):
      experiment_fn = learner.build_experiment_fn(hps, input_md)
      tf.logging.info('Testing train, eval...')
      eval_metrics = experiment_fn(model_dir).test()

      tf.logging.info('Testing model export...')
      export_dir = export.run_export(
          hps, image_shape, model_dir, export_dir_base)
      export.run_test_inference(image_shape, export_dir)

    self.assertIsNotNone(eval_metrics.get('loss'))


if __name__ == '__main__':
  tf.test.main()
