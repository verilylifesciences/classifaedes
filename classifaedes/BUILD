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

# Male / female mosquito classifier for Debug project.

load("@pipdeps//:requirements.bzl", "requirement")

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

py_library(
    name = "simple_image_lib",
    srcs = ["simple_image_lib.py"],
    srcs_version = "PY3",
    deps = [
        requirement("Pillow"),
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "simple_image_lib_test",
    srcs = ["simple_image_lib_test.py"],
    data = ["testdata/raw_data/males/M_00001.bmp"],
    python_version = "PY3",
    deps = [
        ":simple_image_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        requirement("numpy"),
    ],
)

py_library(
    name = "metadata",
    srcs = ["metadata.py"],
    srcs_version = "PY3",
    deps = [
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "learner",
    srcs = ["learner.py"],
    python_version = "PY3",
    deps = [
        ":learner_main_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "learner_main_lib",
    srcs = [
        "hparams_lib.py",
        "inputs_lib.py",
        "learner.py",
        "model_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":metadata",
        requirement("tensorflow"),
        # contrib/framework:framework_py
        # contrib/learn
        # contrib/metrics:metrics_py
        # contrib/slim
        # contrib/slim/python/slim/nets:inception
        # contrib/training:training_py
        
    ],
)

py_test(
    name = "learner_test",
    size = "large",
    srcs = ["learner_test.py"],
    data = glob([
        "testdata/data_dir/test/*",
        "testdata/data_dir/train/*",
    ]) + ["testdata/data_dir/metadata.json"],
    python_version = "PY3",
    deps = [
        ":export_lib",
        ":learner_main_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        requirement("absl-py"), # flagsaver
        requirement("tensorflow"),
    ],
)

py_test(
    name = "hparams_lib_test",
    srcs = ["hparams_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":learner_main_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        requirement("tensorflow"),
    ],
)

py_test(
    name = "inputs_lib_test",
    srcs = ["inputs_lib_test.py"],
    data = glob([
        "testdata/data_dir/test/*",
        "testdata/data_dir/train/*",
    ]) + ["testdata/data_dir/metadata.json"],
    python_version = "PY3",
    deps = [
        ":learner_main_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "model_lib_test",
    size = "large",
    srcs = ["model_lib_test.py"],
    python_version = "PY3",
    deps = [
        ":learner_main_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        requirement("absl-py"), # parameterized
        requirement("tensorflow"),
        # contrib/learn
    ],
)

py_binary(
    name = "export",
    srcs = ["export.py"],
    python_version = "PY3",
    deps = [
        ":export_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "export_lib",
    srcs = ["export.py"],
    srcs_version = "PY3",
    deps = [
        ":learner_main_lib",
        ":metadata",
        ":simple_image_lib",
        requirement("numpy"),
        requirement("tensorflow"),
        # contrib/learn
    ],
)
