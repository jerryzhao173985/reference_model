"""Tests for json2numpy.py."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
import os

import numpy as np
import pytest
from json2numpy.json2numpy import main


@pytest.mark.parametrize(
    "npy_filename,json_filename,data_type",
    [
        ("single_num.npy", "single_num.json", np.int8),
        ("multiple_num.npy", "multiple_num.json", np.int8),
        ("single_num.npy", "single_num.json", np.int16),
        ("multiple_num.npy", "multiple_num.json", np.int16),
        ("single_num.npy", "single_num.json", np.int32),
        ("multiple_num.npy", "multiple_num.json", np.int32),
        ("single_num.npy", "single_num.json", np.int64),
        ("multiple_num.npy", "multiple_num.json", np.int64),
        ("single_num.npy", "single_num.json", np.uint8),
        ("multiple_num.npy", "multiple_num.json", np.uint8),
        ("single_num.npy", "single_num.json", np.uint16),
        ("multiple_num.npy", "multiple_num.json", np.uint16),
        ("single_num.npy", "single_num.json", np.uint32),
        ("multiple_num.npy", "multiple_num.json", np.uint32),
        ("single_num.npy", "single_num.json", np.uint64),
        ("multiple_num.npy", "multiple_num.json", np.uint64),
        #        ("single_num.npy", "single_num.json", np.float16),
        #        ("multiple_num.npy", "multiple_num.json", np.float16),
        ("single_num.npy", "single_num.json", np.float32),
        ("multiple_num.npy", "multiple_num.json", np.float32),
        ("single_num.npy", "single_num.json", np.float64),
        ("multiple_num.npy", "multiple_num.json", np.float64),
        ("single_num.npy", "single_num.json", bool),
        ("multiple_num.npy", "multiple_num.json", bool),
    ],
)
def test_json2numpy_npy_file(npy_filename, json_filename, data_type):
    """Test conversion to JSON."""
    # Generate numpy data.
    if "single" in npy_filename:
        npy_data = np.ndarray(shape=(1, 1), dtype=data_type)
    elif "multiple" in npy_filename:
        npy_data = np.ndarray(shape=(2, 3), dtype=data_type)

    # Get filepaths
    npy_file = os.path.join(os.path.dirname(__file__), npy_filename)
    json_file = os.path.join(os.path.dirname(__file__), json_filename)

    # Save npy data to file and reload it.
    with open(npy_file, "wb") as f:
        np.save(f, npy_data)
    npy_data = np.load(npy_file)

    args = [npy_file]
    """Converts npy file to json"""
    assert main(args) == 0

    json_data = json.load(open(json_file))
    assert np.dtype(json_data["type"]) == npy_data.dtype
    assert np.array(json_data["data"]).shape == npy_data.shape
    assert (np.array(json_data["data"], dtype=data_type) == npy_data).all()

    # Remove files created
    if os.path.exists(npy_file):
        os.remove(npy_file)
    if os.path.exists(json_file):
        os.remove(json_file)


@pytest.mark.parametrize(
    "npy_filename,json_filename,data_type",
    [
        ("single_num.npy", "single_num.json", np.int8),
        ("multiple_num.npy", "multiple_num.json", np.int8),
        ("single_num.npy", "single_num.json", np.int16),
        ("multiple_num.npy", "multiple_num.json", np.int16),
        ("single_num.npy", "single_num.json", np.int32),
        ("multiple_num.npy", "multiple_num.json", np.int32),
        ("single_num.npy", "single_num.json", np.int64),
        ("multiple_num.npy", "multiple_num.json", np.int64),
        ("single_num.npy", "single_num.json", np.uint8),
        ("multiple_num.npy", "multiple_num.json", np.uint8),
        ("single_num.npy", "single_num.json", np.uint16),
        ("multiple_num.npy", "multiple_num.json", np.uint16),
        ("single_num.npy", "single_num.json", np.uint32),
        ("multiple_num.npy", "multiple_num.json", np.uint32),
        ("single_num.npy", "single_num.json", np.uint64),
        ("multiple_num.npy", "multiple_num.json", np.uint64),
        #        ("single_num.npy", "single_num.json", np.float16),
        #        ("multiple_num.npy", "multiple_num.json", np.float16),
        ("single_num.npy", "single_num.json", np.float32),
        ("multiple_num.npy", "multiple_num.json", np.float32),
        ("single_num.npy", "single_num.json", np.float64),
        ("multiple_num.npy", "multiple_num.json", np.float64),
        ("single_num.npy", "single_num.json", bool),
        ("multiple_num.npy", "multiple_num.json", bool),
    ],
)
def test_json2numpy_json_file(npy_filename, json_filename, data_type):
    """Test conversion to binary."""
    # Generate json data.
    if "single" in npy_filename:
        npy_data = np.ndarray(shape=(1, 1), dtype=data_type)
    elif "multiple" in npy_filename:
        npy_data = np.ndarray(shape=(2, 3), dtype=data_type)

    # Generate json dictionary
    list_data = npy_data.tolist()
    json_data_type = str(npy_data.dtype)

    json_data = {}
    json_data["type"] = json_data_type
    json_data["data"] = list_data

    # Get filepaths
    npy_file = os.path.join(os.path.dirname(__file__), npy_filename)
    json_file = os.path.join(os.path.dirname(__file__), json_filename)

    # Save json data to file and reload it.
    with open(json_file, "w") as f:
        json.dump(json_data, f)
    json_data = json.load(open(json_file))

    args = [json_file]
    """Converts json file to npy"""
    assert main(args) == 0

    npy_data = np.load(npy_file)
    assert np.dtype(json_data["type"]) == npy_data.dtype
    assert np.array(json_data["data"]).shape == npy_data.shape
    assert (np.array(json_data["data"], dtype=data_type) == npy_data).all()

    # Remove files created
    if os.path.exists(npy_file):
        os.remove(npy_file)
    if os.path.exists(json_file):
        os.remove(json_file)
