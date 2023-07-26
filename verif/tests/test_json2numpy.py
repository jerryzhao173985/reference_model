"""Tests for json2numpy.py."""
# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
from json2numpy.json2numpy import main


DTYPE_RANGES = {
    np.int8.__name__: [-128, 128],
    np.uint8.__name__: [0, 256],
    np.int16.__name__: [-32768, 32768],
    np.uint16.__name__: [0, 65536],
    np.int32.__name__: [-(1 << 31), (1 << 31)],
    np.uint32.__name__: [0, (1 << 32)],
    np.int64.__name__: [-(1 << 63), (1 << 63)],
    np.uint64.__name__: [0, (1 << 64)],
}


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
        # Not implemented due to json.dump issue
        # ("single_num.npy", "single_num.json", np.uint64),
        # ("multiple_num.npy", "multiple_num.json", np.uint64),
        ("single_num.npy", "single_num.json", np.float16),
        ("multiple_num.npy", "multiple_num.json", np.float16),
        ("single_num.npy", "single_num.json", np.float32),
        ("multiple_num.npy", "multiple_num.json", np.float32),
        ("single_num.npy", "single_num.json", np.float64),
        ("multiple_num.npy", "multiple_num.json", np.float64),
        ("single_num.npy", "single_num.json", bool),
        ("multiple_num.npy", "multiple_num.json", bool),
    ],
)
def test_json2numpy_there_and_back(npy_filename, json_filename, data_type):
    """Test conversion to JSON."""
    # Generate numpy data.
    if "single" in npy_filename:
        shape = (1,)
    elif "multiple" in npy_filename:
        shape = (4, 6, 5)

    rng = np.random.default_rng()
    nan_location = None
    if data_type in [np.float16, np.float32, np.float64]:
        gen_type = np.float32 if data_type == np.float16 else data_type
        generated_npy_data = rng.standard_normal(size=shape, dtype=gen_type).astype(
            data_type
        )
        if len(shape) > 1:
            # Set some NANs and INFs
            nan_location = (1, 2, 3)
            generated_npy_data[nan_location] = np.nan
            generated_npy_data[(3, 2, 1)] = np.inf
            generated_npy_data[(0, 5, 2)] = -np.inf
    elif data_type == bool:
        generated_npy_data = rng.choice([True, False], size=shape).astype(bool)
    else:
        range = DTYPE_RANGES[data_type.__name__]
        generated_npy_data = rng.integers(
            low=range[0], high=range[1], size=shape, dtype=data_type
        )

    # Get filepaths
    npy_file = os.path.join(os.path.dirname(__file__), npy_filename)
    json_file = os.path.join(os.path.dirname(__file__), json_filename)

    # Save npy data to file and reload it.
    with open(npy_file, "wb") as f:
        np.save(f, generated_npy_data)
    npy_data = np.load(npy_file)

    # Test json2numpy - converts npy file to json
    args = [npy_file]
    assert main(args) == 0

    # Remove the numpy file and convert json back to npy
    os.remove(npy_file)
    assert not os.path.exists(npy_file)
    args = [json_file]
    assert main(args) == 0

    converted_npy_data = np.load(npy_file)

    # Check that the original data equals the npy->json->npy data
    assert converted_npy_data.dtype == npy_data.dtype
    assert converted_npy_data.shape == npy_data.shape
    equals = np.equal(converted_npy_data, npy_data)
    if nan_location is not None:
        # NaNs do not usaually equal - so check and set
        if np.isnan(converted_npy_data[nan_location]) and np.isnan(
            npy_data[nan_location]
        ):
            equals[nan_location] = True
    if not np.all(equals):
        print("JSONed:  ", converted_npy_data)
        print("Original:", npy_data)
        print("Equals:  ", equals)
    assert np.all(equals)

    # Remove files created
    if os.path.exists(npy_file):
        os.remove(npy_file)
    if os.path.exists(json_file):
        os.remove(json_file)
