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
"""Tests for hparams_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from classifaedes import hparams_lib
import tensorflow.compat.v1 as tf


class HparamsLibTest(tf.test.TestCase):

  def testIndentedSerialize(self):
    """Tests that our slightly customized serialization can be parsed.

    hparams_lib._human_serialize() uses indented JSON to improve readability.
    """
    hps1 = hparams_lib.defaults()
    serialized = hparams_lib._human_serialize(hps1)

    hps2 = hparams_lib.defaults()
    hps2.parse_json(serialized)

    self.assertDictEqual(hps1.values(), hps2.values())


if __name__ == '__main__':
  tf.test.main()
