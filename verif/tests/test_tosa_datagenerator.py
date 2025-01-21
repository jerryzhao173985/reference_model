"""Tests for the python interface to the data generator library."""
# Copyright (c) 2023,2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest
from generator.datagenerator import GenerateError
from generator.datagenerator import GenerateLibrary
from test_utils import GENERATE_LIB_PATH

# NOTE: These tests are marked as POST COMMIT
# To run them, please build the reference_model in a local "build" directory
# (as per the README) and run them using: pytest -m "postcommit"

TEST_DIR = Path(__file__).parent


@pytest.mark.postcommit
def test_generate_lib_built():
    """First test to check the library has been built."""
    assert GENERATE_LIB_PATH.is_file()


@pytest.mark.postcommit
def test_checker_generate_load_fail():
    with pytest.raises(GenerateError) as excinfo:
        GenerateLibrary(Path("/place-that-does-not-exist"))
    assert str(excinfo.value).startswith("Could not find generate library")


@pytest.mark.postcommit
def test_checker_generate_load():
    glib = GenerateLibrary(GENERATE_LIB_PATH)
    assert glib


JSON_DATAGEN_DOT_PRODUCT = {
    "tosa_file": "test.json",
    "ifm_name": ["input-0", "input-1"],
    "ifm_file": ["input-0.npy", "input-1.npy"],
    "ofm_name": ["result-0"],
    "ofm_file": ["result-0.npy"],
    "meta": {
        "data_gen": {
            "version": "0.1",
            "tensors": {
                "input-0": {
                    "generator": "DOT_PRODUCT",
                    "data_type": "FP32",
                    "input_type": "VARIABLE",
                    "shape": [3, 5, 4],
                    "input_pos": 0,
                    "op": "MATMUL",
                    "dot_product_info": {
                        "s": 0,
                        "ks": 4,
                        "acc_type": "FP32",
                        "otherInputType": "FP32",
                    },
                },
                "input-1": {
                    "generator": "DOT_PRODUCT",
                    "data_type": "FP32",
                    "input_type": "VARIABLE",
                    "shape": [3, 4, 6],
                    "input_pos": 1,
                    "op": "MATMUL",
                    "dot_product_info": {
                        "s": 0,
                        "ks": 4,
                        "acc_type": "FP32",
                        "otherInputType": "FP32",
                    },
                },
            },
        }
    },
}


@pytest.mark.postcommit
def test_generate_dot_product_check():
    glib = GenerateLibrary(GENERATE_LIB_PATH)
    assert glib

    json_config = JSON_DATAGEN_DOT_PRODUCT
    glib.set_config(json_config)

    glib.write_numpy_files(TEST_DIR)

    # Test the files exist and are the expected numpy files
    for f, n in zip(json_config["ifm_file"], json_config["ifm_name"]):
        file = TEST_DIR / f
        assert file.is_file()
        arr = np.load(file)
        assert arr.shape == tuple(
            json_config["meta"]["data_gen"]["tensors"][n]["shape"]
        )
        assert arr.dtype == np.float32
        file.unlink()


@pytest.mark.postcommit
def test_generate_dot_product_check_fail_names():
    glib = GenerateLibrary(GENERATE_LIB_PATH)
    assert glib

    # Fix up the JSON to have the wrong names
    json_config = JSON_DATAGEN_DOT_PRODUCT.copy()
    json_config["ifm_name"] = ["not-input0", "not-input1"]
    glib.set_config(json_config)

    with pytest.raises(GenerateError) as excinfo:
        glib.write_numpy_files(TEST_DIR)
    info = str(excinfo.value).split("\n")
    for i, n in enumerate(json_config["ifm_name"]):
        assert info[i].startswith(f"ERROR: Failed to create data for tensor {n}")

    for f in json_config["ifm_file"]:
        file = TEST_DIR / f
        assert not file.is_file()


@pytest.mark.postcommit
def test_generate_tensor_data_check():
    glib = GenerateLibrary(GENERATE_LIB_PATH)
    assert glib

    json_config = JSON_DATAGEN_DOT_PRODUCT["meta"]["data_gen"]

    for n in JSON_DATAGEN_DOT_PRODUCT["ifm_name"]:
        arr = glib.get_tensor_data(n, json_config)

        assert arr.shape == tuple(json_config["tensors"][n]["shape"])
        assert arr.dtype == np.float32
