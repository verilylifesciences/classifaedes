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
"""Tests for model lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import parameterized
from classifaedes import hparams_lib
from classifaedes import model_lib
import tensorflow.compat.v1 as tf
from tensorflow.contrib import learn as contrib_learn


@parameterized.named_parameters(
    ('Defaults', hparams_lib.defaults()),
    ('InceptionV3b', hparams_lib.defaults().override_from_dict({
        'arch': 'inception_v3b'
    })),
)
class ModelLibTest(tf.test.TestCase):

  def setUp(self):
    super(ModelLibTest, self).setUp()
    self._input_md = {
        'num_examples_negative': 182580,
        'num_examples_positive': 2050118,
    }

  def _input_fn(self, batch_size):
    targets = tf.random_uniform([batch_size], dtype=tf.int32, maxval=2, seed=1)
    images = tf.random_uniform([batch_size, 120, 130, 1])
    return {'images': images}, tf.equal(1, targets)

  def testBuildTrainGraph(self, hps):
    batch_size = hps.batch_size
    with tf.Graph().as_default():
      inputs, targets = self._input_fn(batch_size)
      model_fn = model_lib.build_model_fn(hps, self._input_md)

      probabilities, loss, train_op = model_fn(inputs, targets,
                                               contrib_learn.ModeKeys.TRAIN)

    self.assertEqual(probabilities['outputs'].dtype, tf.float32)
    self.assertEqual(loss.dtype, tf.float32)
    self.assertIsNotNone(train_op)

    self.assertEqual(probabilities['outputs'].shape, [batch_size])
    self.assertEqual(loss.shape, [])

  def testBuildEvalGraph(self, hps):
    batch_size = hps.batch_size
    with tf.Graph().as_default():
      inputs, targets = self._input_fn(batch_size)
      model_fn = model_lib.build_model_fn(hps, self._input_md)

      probabilities, loss, train_op = model_fn(inputs, targets,
                                               contrib_learn.ModeKeys.EVAL)

    self.assertEqual(probabilities['outputs'].dtype, tf.float32)
    self.assertEqual(loss.dtype, tf.float32)
    self.assertIsNone(train_op)

    self.assertEqual(probabilities['outputs'].shape, [batch_size])
    self.assertEqual(loss.shape, [])

  def testRunTrainGraph(self, hps):
    with self.test_session() as sess:
      inputs, targets = self._input_fn(hps.batch_size)
      model_fn = model_lib.build_model_fn(hps, self._input_md)

      probabilities_tensor, loss_tensor, train_op = model_fn(
          inputs, targets, contrib_learn.ModeKeys.TRAIN)

      tf.global_variables_initializer().run()
      sess.run(train_op)
      sess.run([probabilities_tensor, loss_tensor])

  def testRunEvalGraph(self, hps):
    with self.test_session() as sess:
      inputs, targets = self._input_fn(hps.batch_size)
      model_fn = model_lib.build_model_fn(hps, self._input_md)

      probabilities_tensor, loss_tensor, _ = model_fn(
          inputs, targets, contrib_learn.ModeKeys.EVAL)

      tf.global_variables_initializer().run()
      sess.run([probabilities_tensor, loss_tensor])


if __name__ == '__main__':
  tf.test.main()
