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
"""Tests model's input_fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path



from classifaedes import hparams_lib
from classifaedes import inputs_lib
from classifaedes import metadata
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# From testdata/data_dir/metadata.json.
MAX_HEIGHT = 346
MAX_WIDTH = 672


class InputsLibTest(tf.test.TestCase):

  def setUp(self):
    super(InputsLibTest, self).setUp()
    self.data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        
        'testdata/data_dir')

  def testInputsGraph(self):
    hps = hparams_lib.defaults()
    hps.batch_size = 19
    input_md = metadata.load_metadata(data_dir=self.data_dir)

    input_dict, labels = inputs_lib.build_input_fn(
        hps, input_md, self.data_dir, 'train')()

    self.assertSameElements(['images'], input_dict.keys())
    images = input_dict['images']

    self.assertEqual(labels.dtype, tf.bool)
    self.assertEqual(labels.shape, [hps.batch_size])

    self.assertEqual(images.dtype, tf.float32)
    self.assertEqual(images.shape,
                     [hps.batch_size, MAX_HEIGHT, MAX_WIDTH, 1])

  def testRunInputFn(self):
    hps = hparams_lib.defaults()
    hps.batch_size = 2
    input_md = metadata.load_metadata(data_dir=self.data_dir)

    tf_input_dict, tf_labels = inputs_lib.build_input_fn(
        hps, input_md, self.data_dir, 'train')()

    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      images, labels = sess.run([tf_input_dict['images'], tf_labels])

      self.assertEqual(images.dtype, np.float32)
      self.assertEqual(images.shape, (2, MAX_HEIGHT, MAX_WIDTH, 1))
      self.assertEqual(labels.dtype, np.bool)
      self.assertEqual(labels.shape, (2,))

      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  tf.test.main()
