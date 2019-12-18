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
"""A library of very simple image processing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io


import numpy as np
from PIL import Image as pil_image

import tensorflow.compat.v1 as tf
gfile = tf.gfile


def encode_png(image):
  buf = io.BytesIO()
  pil_image.fromarray(image).save(buf, format='PNG')
  return buf.getvalue()


def decode_image(encoded):
  buf = io.BytesIO(encoded)
  return np.array(pil_image.open(buf))


def download_image(image_path):
  with gfile.Open(image_path, 'rb') as f:
    try:
      image = pil_image.open(f).convert('L')
    except IOError as e:
      # Raise a ValueError if the image data is invalid
      raise ValueError(e)
    return np.array(image)
