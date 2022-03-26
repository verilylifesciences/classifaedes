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
"""Learner for Debug male/female classification model."""

from __future__ import absolute_import
from __future__ import division

import functools
import json
import time
import warnings



from classifaedes import hparams_lib
from classifaedes import inputs_lib
from classifaedes import metadata
from classifaedes import model_lib
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import learn as contrib_learn

# Schenanegans to import extra tensforflow dependencies that are not
# pulled in by default in Google's system.
import tensorflow.contrib.learn as tflearn

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('hparams', '', 'Hyperparameters. See hparams_lib.py.')
flags.DEFINE_string('data_dir', None,
                    'Input data dir with train, test subdirs.')

flags.DEFINE_integer('save_summary_steps', 20, 'Chief summary-save period.')
flags.DEFINE_integer('save_checkpoints_secs', 60, 'Checkpoint-save period.')
flags.DEFINE_integer('evals_per_ckpt', 20,
                     'Number of batches to eval per checkpoint.')
flags.DEFINE_integer('train_steps', None, 'Train until this global-step.')

# Note: --eval_delay_secs is a deprecated flag in a tf-learn dependency.
flags.DEFINE_integer('eval_delay_seconds', 30,
                     'Delay continuous eval for this many seconds.')
flags.DEFINE_integer('worker_stagger_seconds', 20,
                     'Delay worker startup by N * task_num seconds.')


# Notable libraries that define additional flags:
# * tf/contrib/learn/learn_runner.py

CHIEF_SCHEDULES = ['train', 'train_and_evaluate', 'local_run']


def _get_hps():
  """Returns HParams from flags."""
  hps = hparams_lib.defaults()
  hps.parse(FLAGS.hparams)

  tf.logging.info('HParams:\n%s', json.dumps(hps.values(), indent=2))
  return hps


def _prep_model_dir(hps, input_md):
  model_dir = FLAGS.output_dir
  tf.gfile.MakeDirs(model_dir, mode=0o775)

  hparams_lib.write_to_file(hps, model_dir)

  md_path = metadata.metadata_path(model_dir=model_dir)
  with tf.gfile.Open(md_path, 'w') as fl:
    json.dump(input_md, fl)


def build_experiment_fn(hps, input_md):
  """Builds experiment_fn for use with learn_runner."""

  def experiment_fn(output_dir):
    """Returns tf.learn Experiment for the model."""

    estimator = contrib_learn.Estimator(
        model_fn=model_lib.build_model_fn(hps, input_md),
        model_dir=output_dir,
        config=tf_estimator.RunConfig(
            save_summary_steps=FLAGS.save_summary_steps,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        ),
    )

    data_dir = FLAGS.data_dir
    train_input_fn = inputs_lib.build_input_fn(hps, input_md, data_dir, 'train')
    eval_input_fn = inputs_lib.build_input_fn(hps, input_md, data_dir, 'test')

    experiment_partial = functools.partial(
        contrib_learn.Experiment,
        eval_steps=FLAGS.evals_per_ckpt,
    )

    return experiment_partial(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        eval_metrics=model_lib.build_eval_metrics(),
        train_steps=(FLAGS.train_steps or None),
        continuous_eval_throttle_secs=3,
        eval_delay_secs=FLAGS.eval_delay_seconds,
    )
  return experiment_fn


def main(unused_argv):
  warnings.simplefilter('error')
  # Stagger worker startups for stability.
  time.sleep(FLAGS.worker_stagger_seconds * FLAGS.brain_task)

  hps = _get_hps()

  input_md = metadata.load_metadata(data_dir=FLAGS.data_dir)

  _prep_model_dir(hps, input_md)

  tflearn.learn_runner.run(build_experiment_fn(hps, input_md))


if __name__ == '__main__':
  tf.app.run()
