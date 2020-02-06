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
"""Tests for simple_image_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path



from classifaedes import simple_image_lib as sil
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class SimpleImageLibTest(tf.test.TestCase):

  def testEncodeDecodeInversion(self):
    image = np.arange(10 * 11 * 3, dtype=np.uint8).reshape([10, 11, 3])
    encoded = sil.encode_png(image)
    decoded = sil.decode_image(encoded)
    re_encoded = sil.encode_png(decoded)

    self.assertEqual(image.dtype, decoded.dtype)
    self.assertEqual(image.shape, decoded.shape)
    self.assertTrue((image == decoded).all())
    self.assertEqual(encoded, re_encoded)

  def testDownloadImage(self):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        
                        'testdata/raw_data/males/M_00001.bmp')
    image = sil.download_image(path)
    self.assertEqual(np.uint8, image.dtype)
    self.assertEqual((311, 453), image.shape)


if __name__ == '__main__':
  tf.test.main()
