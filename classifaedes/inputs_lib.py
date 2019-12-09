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
"""Builds the model input_fn for training and eval."""

from __future__ import absolute_import
from __future__ import division

import functools
import os.path
import random



from classifaedes import metadata
import tensorflow as tf
from tensorflow.contrib import slim

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_list('positive_labels', ['males', 'multiple-males'],
                  'Comma separated list of positive labels.')

flags.DEFINE_integer('eval_batch_size', 32,
                     'Batch size for eval (for consistent tuning eval).')

flags.DEFINE_integer('read_q_capacity', 16, 'Read queue capacity.')
flags.DEFINE_integer('read_q_threads', 2, 'Number of data-read threads.')

flags.DEFINE_integer('shuffle_q_capacity', 512, 'Shuffle queue capacity.')
flags.DEFINE_integer('shuffle_q_min_after_deq', 128,
                     'Minimum number of examples in the shufle queue.')
flags.DEFINE_integer('shuffle_q_threads', 4,
                     'Number of queue runner threads for the shuffle queue. '
                     'These threads perform image pre-processing.')


def prep_image(img, image_shape, is_training=False):
  """Perform image preprocessing for training and serving."""
  h, w = image_shape
  img = tf.image.convert_image_dtype(img, tf.float32)
  if is_training:
    img = _distort_image(img)
  # Distort after padding to avoid revealing the distortion effects in padding.
  img = tf.image.pad_to_bounding_box(img, 0, 0, h, w)
  return img


def build_input_fn(hps, input_md, data_dir, split):
  """Returns input_fn for tf.learn Estimator.

  Args:
    hps: HParams.
    input_md: Input metadata. See metadata.py for details.
    data_dir: Path to directory containing train, test data.
    split: "train" or "test".

  Returns:
    Estimator input_fn - callable input graph builder.
  """
  assert split in ('train', 'test')
  filepath = os.path.join(data_dir, split, 'Examples-?????-of-?????')
  decoder = _build_decoder()
  image_shape = metadata.shape_from_metadata(input_md)
  tf.logging.info('Using image shape: %s', image_shape)

  is_training = split == 'train'

  def input_fn():
    """Builds input ops for the train / eval graph.

    Returns:
    A 2-tuple of (inputs, targets).
      inputs: Dictionary of input Tensors.
        * images: Float Tensor of shape [N, H, W, 1].
      targets: Boolean Tensor of shape [N].
    """
    data = slim.dataset.Dataset(
        data_sources=[filepath],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions={
            'image': 'Grayscale image.',
            'label': 'String label.',
        },
    )
    dataprovider = slim.dataset_data_provider.DatasetDataProvider(
        data,
        num_readers=FLAGS.read_q_threads,
        common_queue_capacity=FLAGS.read_q_capacity,
        common_queue_min=(FLAGS.read_q_capacity // 2),
        shuffle=is_training,
    )
    img, str_label = dataprovider.get(['image', 'label'])

    label = tf.reduce_any([tf.equal(s, str_label)
                           for s in FLAGS.positive_labels], axis=0)

    img = prep_image(img, image_shape, is_training=is_training)
    img, label = _batch_examples(hps, [img, label], is_training)
    tf.logging.error('Image shape: %s', img.get_shape().as_list())
    tf.summary.image('positives', tf.boolean_mask(img, label))
    tf.summary.image('negatives', tf.boolean_mask(img, tf.logical_not(label)))
    return {'images': img}, label

  return input_fn

#
# Private functions.
#


def _build_decoder():
  """Builds the TFExampleDecoder for reading train / eval data."""
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/format': tf.FixedLenFeature([], tf.string, default_value='png'),
      'label': tf.FixedLenFeature([], tf.string),
  }
  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(channels=1),
      'label': slim.tfexample_decoder.Tensor('label'),
  }
  return slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                 items_to_handlers)


def _batch_examples(hps, tensors, is_training):
  """Enqueue preprocessed input tensors for batching ahead of training path."""
  batch_size = FLAGS.eval_batch_size
  batch_fn = tf.train.batch
  # For eval, use deterministic batching and fixed batch_size.
  if is_training:
    batch_size = hps.batch_size
    batch_fn = functools.partial(
        tf.train.shuffle_batch,
        min_after_dequeue=FLAGS.shuffle_q_min_after_deq)

  return batch_fn(
      tensors,
      batch_size=batch_size,
      capacity=FLAGS.shuffle_q_capacity,
      num_threads=FLAGS.shuffle_q_threads,
  )


def _distort_image(img):
  """Randomly distort the image."""
  with tf.name_scope('image_distortions'):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    pixel_xforms = [
        functools.partial(tf.image.random_brightness, max_delta=0.3),
        functools.partial(tf.image.random_contrast, lower=0.5, upper=1.5)
    ]
    random.shuffle(pixel_xforms)
    for xform in pixel_xforms:
      img = xform(img)
    return tf.clip_by_value(img, 0.0, 1.0)
