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
"""Model and eval graph builder functions."""
from __future__ import absolute_import
from __future__ import division

import functools as ft



from classifaedes import metadata
import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import learn as contrib_learn
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import slim

# Schenanegans to import extra tensforflow dependencies that are not
# pulled in by default in Google's system.
from tensorflow.contrib.slim.nets import inception

tfmetrics = contrib_metrics


def build_model_fn(hps, input_md):
  """Returns the Estimator model_fn given HParams.

  Args:
    hps: The HParams to use.
    input_md: Input Metadata dictionary. See metadata.py for expected fields.
  """

  def model_fn(inputs, targets, mode):
    """Builds the inference or train graph.

    Args:
      inputs: Dictionary of input Tensors.
        * images: Float Tensor of shape [N, H, W, 1].
      targets: Boolean Tensor of shape [N].
      mode: tf.contrib.learn.ModeKeys.(TRAIN|EVAL).

    Returns:
      predictions: Float Tensor batched probability predictions (shape=[N]).
      loss: Scalar loss Tensor.
      train_op: Op/Tensor that triggers gradient updates (None when mode=EVAL).
    """
    imgs = inputs['images']
    is_training = mode == contrib_learn.ModeKeys.TRAIN

    logits = _build_inference_net(hps, imgs, is_training)

    loss = None
    if mode != contrib_learn.ModeKeys.INFER:
      if hps.use_global_objective_recall_at_precision:
        raise NotImplementedError(
            'Global Objective Optimization is not Available')
      else:
        if (metadata.NUM_EXAMPLES_POS in input_md
            and metadata.NUM_EXAMPLES_POS in input_md):
          pos_weight = (input_md[metadata.NUM_EXAMPLES_NEG] /
                        input_md[metadata.NUM_EXAMPLES_POS])
          loss_weights = tf.where(targets,
                                  tf.constant(pos_weight,
                                              dtype=tf.float32,
                                              shape=targets.shape),
                                  tf.ones(targets.shape))
        else:
          loss_weights = 1.0

        tf.losses.sigmoid_cross_entropy(targets, logits, loss_weights)
      loss = tf.losses.get_total_loss()
      tf.summary.scalar('loss', loss)

    train_op = None
    if is_training:
      train_op = _build_train_op(hps, loss)
      _log_trainable_vars()
      _summarize_trainable_vars()

    _add_saver()

    return {'outputs': tf.nn.sigmoid(logits)}, loss, train_op

  return model_fn


def build_eval_metrics():
  """Builds dictionary of MetricSpecs for tf.learn's Estimator."""
  metrics = {
      'eval/auc': contrib_learn.MetricSpec(tfmetrics.streaming_auc),
  }

  for k in range(1, 4):
    sens_at_spec = ft.partial(tfmetrics.streaming_sensitivity_at_specificity,
                              specificity=(1.- 0.1**k))
    spec_at_sens = ft.partial(tfmetrics.streaming_specificity_at_sensitivity,
                              sensitivity=(1.- 0.1**k))
    metrics['eval/sens@spec/%d_9s' % k] = contrib_learn.MetricSpec(
        sens_at_spec)
    metrics['eval/spec@sens/%d_9s' % k] = contrib_learn.MetricSpec(
        spec_at_sens)

  for k in range(1, 4):
    sens_at_spec = ft.partial(tfmetrics.streaming_sensitivity_at_specificity,
                              specificity=(1.- 0.5**k))
    spec_at_sens = ft.partial(tfmetrics.streaming_specificity_at_sensitivity,
                              sensitivity=(1.- 0.5**k))
    metrics['eval/sens@spec/%d_1s' % k] = contrib_learn.MetricSpec(
        sens_at_spec)
    metrics['eval/spec@sens/%d_1s' % k] = contrib_learn.MetricSpec(
        spec_at_sens)

  return metrics


def _add_saver():
  assert not tf.get_collection(tf.GraphKeys.SAVERS)
  tf.logging.info('Creating tf.Saver.')
  saver = tf.train.Saver(
      keep_checkpoint_every_n_hours=1,
  )
  tf.add_to_collection(tf.GraphKeys.SAVERS, saver)


def _build_train_op(hps, loss):
  learning_rate = tf.train.exponential_decay(
      hps.lr_init,
      contrib_framework.get_or_create_global_step(),
      decay_steps=hps.lr_decay_steps,
      decay_rate=0.95)
  tf.summary.scalar('learning_rate', learning_rate)
  opt = tf.train.AdamOptimizer(learning_rate, epsilon=hps.adam_epsilon)
  return slim.learning.create_train_op(
      loss, opt,
      clip_gradient_norm=hps.clip_gradient_norm,
      summarize_gradients=True,
  )


def _log_trainable_vars():
  total = 0
  for var in tf.trainable_variables():
    num_params = np.prod(var.get_shape().as_list())
    total += num_params
    tf.logging.info('trainable var; size=%d, shape=%s, name=%s',
                    num_params, var.get_shape().as_list(), var.name)
  tf.logging.info('Total num params: %d', total)


def _summarize_trainable_vars():
  for var in tf.trainable_variables():
    tf.summary.histogram('vars/' + var.name, var)


def _build_inference_net(hps, images, is_training):
  """Build inference network architecture up to logits layer."""
  tf.logging.error('Image shape: %s', images.get_shape().as_list())
  if hps.arch == 'inception_v3a':
    inception_scope = inception.inception_v3_arg_scope()
    with slim.arg_scope(inception_scope):
      logits, _ = inception.inception_v3(
          images,
          dropout_keep_prob=(1.0 - hps.p_dropout),
          depth_multiplier=hps.filter_depth_mul,
          num_classes=1,
          is_training=is_training,
          spatial_squeeze=False)
  elif hps.arch == 'inception_v3b':
    logits, _ = inception.inception_v3(
        images,
        dropout_keep_prob=(1.0 - hps.p_dropout),
        depth_multiplier=hps.filter_depth_mul,
        num_classes=1,
        is_training=is_training,
        spatial_squeeze=False)
  else:
    raise ValueError('Invalid architecture hparams.arch="%s"' % hps.arch)

  logits = tf.squeeze(logits[:, :, 0], [-1, -2])
  tf.summary.scalar('logits_frac_zero', tf.nn.zero_fraction(logits))
  tf.summary.histogram('logits', logits)
  return logits
