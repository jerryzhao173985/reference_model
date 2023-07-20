# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json

# Used by basic_test_generator to create test description


def write_test_json(
    filename,
    tf_model_filename=None,
    tf_result_npy_filename=None,
    tf_result_name=None,
    tflite_model_filename=None,
    tflite_result_npy_filename=None,
    tflite_result_name=None,
    ifm_name=None,
    ifm_file=None,
    ifm_shape=None,
    framework_exclusions=None,
    quantized=False,
    test_name=None,
):

    test_desc = dict()

    if test_name:
        test_desc["name"] = test_name

    if tf_model_filename:
        test_desc["tf_model_filename"] = tf_model_filename

    if tf_result_npy_filename:
        test_desc["tf_result_npy_filename"] = tf_result_npy_filename

    if tf_result_name:
        test_desc["tf_result_name"] = tf_result_name

    if tflite_model_filename:
        test_desc["tflite_model_filename"] = tflite_model_filename

    if tflite_result_npy_filename:
        test_desc["tflite_result_npy_filename"] = tflite_result_npy_filename

    if tflite_result_name:
        test_desc["tflite_result_name"] = tflite_result_name

    if ifm_file:
        if not isinstance(ifm_file, list):
            ifm_file = [ifm_file]
        test_desc["ifm_file"] = ifm_file

    # Make sure these arguments are wrapped as lists
    if ifm_name:
        if not isinstance(ifm_name, list):
            ifm_name = [ifm_name]
        test_desc["ifm_name"] = ifm_name

    if ifm_shape:
        if not isinstance(ifm_shape, list):
            ifm_shape = [ifm_shape]
        test_desc["ifm_shape"] = ifm_shape

    # Some tests cannot be used with specific frameworks.
    # This list indicates which tests should be excluded from a given framework.
    if framework_exclusions:
        if not isinstance(framework_exclusions, list):
            framework_exclusions = [framework_exclusions]
        test_desc["framework_exclusions"] = framework_exclusions

    if quantized:
        test_desc["quantized"] = 1

    with open(filename, "w") as f:
        json.dump(test_desc, f, indent="  ")
