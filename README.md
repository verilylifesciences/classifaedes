# Debug project male/female mosquito classifier

[TOC]

## Introduction

This code is used by the [Debug](https://www.debug.com) project in order to classify male and female Aedes Aegypti mosquitoes for release in an IIT program. Images of the mosquitoes are taken in mechancial sorters and then classified on the likelihood of the image being of a male and thus can be accepted for release.

## Data Preparation
Training and eval data is stored in TFRecord's of `tf.Example` messages in two
directories (train and test). Each `tf.Example` must have the following items:

  *  image/encoded - A greyscale encoded image.
  *  image/format - The format that the image is encoded in (e.g. png)
  *  label - A string specifying the label for the image.


## Training

To test locally, modify the input/output folders appropriately, and run

```shell
bazel -c opt learner -- \
    --schedule='train_and_evaluate' \
    --hparams='' \
    --output_dir=$dev_model_dir \
    --data_dir=$data_dir \
    --read_q_capacity=128 --shuffle_q_capacity=128 --shuffle_q_threads=2 \
    --save_summary_steps=5 \
    --save_checkpoints_secs=60 \
    --evals_per_ckpt=3 \
    --eval_delay_seconds=1 \
    --logtostderr
```

## Inference
For inference in a production environment, first export in SavedModel format. Unless
otherwise specified, the export script writes to `$model_dir/export_`.

```shell
bazel run export -- --model_dir=$model_dir --logtostderr
```

Then, SavedModel can be used for inference with [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) or the [GCP ML Engine](https://cloud.google.com/ml-engine/docs/deploying-models).
