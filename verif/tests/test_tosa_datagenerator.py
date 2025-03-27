"""Tests for the python interface to the data generator library."""
# Copyright (c) 2023-2025, ARM Limited.
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

# Generator types
GEN_PR = "PSEUDO_RANDOM"


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
                        "other_data_type": "FP32",
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
                        "other_data_type": "FP32",
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


def get_json_config(data_type, gen_type, shape_size, data_range=None):
    """Helper function to return data gen config."""
    json_tensor = {}
    json_tensor["generator"] = gen_type
    json_tensor["data_type"] = data_type
    json_tensor["input_type"] = "CONSTANT"
    json_tensor["shape"] = [shape_size]
    json_tensor["input_pos"] = 0
    json_tensor["op"] = "CONST"
    if gen_type == GEN_PR:
        json_tensor["pseudo_random_info"] = {
            "rng_seed": 100,
        }
        if data_range:
            json_tensor["pseudo_random_info"]["range"] = [str(x) for x in data_range]
    else:
        assert False, "Unsupported generator type"

    return {"tensors": {"test": json_tensor}, "version": "0"}


# data_type is the data type to test
# gen_type is the generator mode to test
# data_range is the range (low, high) of values the generator should produce
# NOTE: Main aim here is to produce either all positive or negative values to test
# signed/unsigned conversion through the stack
@pytest.mark.postcommit
@pytest.mark.parametrize(
    "data_type,gen_type,data_range",
    [
        ("INT48", GEN_PR, ((1 << 47) - 10, ((1 << 47) - 1))),
        ("INT48", GEN_PR, (-20, -1)),
        ("INT4", GEN_PR, (1, 7)),
        ("INT4", GEN_PR, (-7, -1)),
        ("BOOL", GEN_PR, (0, 1)),
        ("INT32", GEN_PR, (-20, -1)),
        ("INT8", GEN_PR, (120, 127)),
        ("INT8", GEN_PR, (-128, -120)),
        ("INT16", GEN_PR, (-32768, -32760)),
    ],
)
def test_datagen_lib(data_type, gen_type, data_range):
    """Tests the full data gen round trip."""

    shape_size = 20
    json_config = get_json_config(data_type, gen_type, shape_size, data_range)

    dglib = GenerateLibrary(GENERATE_LIB_PATH)

    arr = dglib.get_tensor_data("test", json_config)

    assert len(arr) == shape_size

    print(arr)

    for val in arr:
        assert val >= data_range[0] and val <= data_range[1]


# data_type is the data type to test
# gen_type is the generator mode to test
# data_in must be in the buffer storage format passed to the datagen library
# data_out is the expected numpy values after conversion
@pytest.mark.postcommit
@pytest.mark.parametrize(
    "data_type,gen_type,data_in,data_out",
    [
        (
            "INT48",
            GEN_PR,
            (
                0xFF,  # Max positive
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x7F,
                0xFF,  # -1
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xDE,  # -34
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0x22,  # 34
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
            ),
            ((1 << 47) - 1, -1, -34, 34),
        ),
        ("INT4", GEN_PR, (0b11100001, 0b00001001), (1, -2, -7)),
        ("BOOL", GEN_PR, (0, 1), (0, 1)),
        ("INT32", GEN_PR, (0xFFFFFFFF, 0x7FFFFFFF), (-1, ((1 << 31) - 1))),
        ("INT8", GEN_PR, (0xFF, 0x7F, 0x03), (-1, 127, 3)),
    ],
)
def test_datagen_conversion(monkeypatch, data_type, gen_type, data_in, data_out):
    """Tests the python conversion back to numpy format."""

    def mockDataGen(self, json_config, tensor_name, buffer, size):
        print("Mocked", data_type)
        for idx, val in enumerate(data_in):
            buffer[idx] = val
        return True

    json_config = get_json_config(data_type, gen_type, len(data_out))

    dglib = GenerateLibrary(GENERATE_LIB_PATH)

    monkeypatch.setattr(GenerateLibrary, "_call_data_gen_library", mockDataGen)

    arr = dglib.get_tensor_data("test", json_config)

    assert len(arr) == len(data_out)

    print(arr)

    for idx, val in enumerate(data_out):
        assert val == arr[idx]


def get_desc_json_config(data_type, file_name, shape_size):
    """Helper function to return desc.json config."""
    desc_json = {}
    desc_json["ifm_name"] = ["test"]
    desc_json["ifm_file"] = [file_name]
    desc_json["meta"] = {"data_gen": get_json_config(data_type, GEN_PR, shape_size)}
    desc_json["ofm_name"] = ["dummy"]
    desc_json["ofm_file"] = ["dummy_file.npy"]
    desc_json["tosa_file"] = "dummy.tosa"

    return desc_json


# dtype_name is the data type to test
# npy_type is the numpy data type expected
# NOTE: Main aim here is to test the save/loading of numpy files
@pytest.mark.postcommit
@pytest.mark.parametrize(
    "dtype_name, npy_type",
    [
        ("BOOL", bool),
        ("INT4", np.int8),
        ("INT8", np.int8),
        ("INT16", np.int16),
        ("INT32", np.int32),
        ("INT48", np.int64),
        ("FP32", np.float32),
        ("FP16", np.float16),
        ("BF16", np.dtype("V2")),
        ("FP8E4M3", np.dtype("V1")),
        ("FP8E5M2", np.uint8),
        ("SHAPE", np.int64),
    ],
)
def test_datagen_save(dtype_name, npy_type):
    """Tests the saving/loading of numpy."""
    out_file = f"dg_output_{dtype_name}.npy"
    shape_size = 13
    desc_json = get_desc_json_config(dtype_name, out_file, shape_size)

    dglib = GenerateLibrary(GENERATE_LIB_PATH)
    dglib.set_config(desc_json)
    dglib.write_numpy_files(TEST_DIR)

    file = TEST_DIR / out_file
    assert file.is_file()
    arr = np.load(file)
    assert arr.shape == (shape_size,)
    assert arr.dtype == npy_type
    file.unlink()
