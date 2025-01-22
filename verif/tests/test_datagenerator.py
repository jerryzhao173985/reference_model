"""Tests for datagenerator"""
# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import conformance.model_files as cmf
import pytest
from generator.datagenerator import GenerateLibrary


# Location of reference model binaries
REF_MODEL_DIR = Path(__file__).resolve().parents[2]
REF_MODEL_EXE_PATH = cmf.find_tosa_file(
    cmf.TosaFileType.REF_MODEL, REF_MODEL_DIR, False
)
GENERATE_LIB_PATH = cmf.find_tosa_file(
    cmf.TosaFileType.GENERATE_LIBRARY, REF_MODEL_EXE_PATH
)

# Generator types
GEN_PR = "PSEUDO_RANDOM"


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


# NOTE: These tests are marked as POST COMMIT
# To run them, please build the reference_model in a local "build" directory
# (as per the README) and run them using: pytest -m "postcommit"


@pytest.mark.postcommit
def test_generate_lib_built():
    """First test to check the generate lib has been built."""
    assert (
        GENERATE_LIB_PATH.is_file()
    ), "Generate library needed for testing has not been found"


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
