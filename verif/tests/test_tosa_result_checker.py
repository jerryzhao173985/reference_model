"""Tests for tosa_result_checker.py."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import checker.tosa_result_checker as trc
import numpy as np
import pytest


def _create_data_file(name, npy_data):
    """Create numpy data file."""
    file = Path(__file__).parent / name
    with open(file, "wb") as f:
        np.save(f, npy_data)
    return file


def _create_empty_file(name):
    """Create numpy data file."""
    file = Path(__file__).parent / name
    f = open(file, "wb")
    f.close()
    return file


def _delete_data_file(file: Path):
    """Delete numpy data file."""
    file.unlink()


@pytest.mark.parametrize(
    "data_type,expected",
    [
        (np.int8, trc.TestResult.MISMATCH),
        (np.int16, trc.TestResult.MISMATCH),
        (np.int32, trc.TestResult.PASS),
        (np.int64, trc.TestResult.PASS),
        (np.uint8, trc.TestResult.MISMATCH),
        (np.uint16, trc.TestResult.MISMATCH),
        (np.uint32, trc.TestResult.MISMATCH),
        (np.uint64, trc.TestResult.MISMATCH),
        (np.float16, trc.TestResult.PASS),
        (np.float32, trc.TestResult.PASS),
        (np.float64, trc.TestResult.MISMATCH),
        (bool, trc.TestResult.PASS),
    ],
)
def test_supported_types(data_type, expected):
    """Check which data types are supported."""
    # Generate data
    npy_data = np.ndarray(shape=(2, 3), dtype=data_type)

    # Save data as reference and result files to compare.
    reference_file = _create_data_file("reference.npy", npy_data)
    result_file = _create_data_file("result.npy", npy_data)

    args = [str(reference_file), str(result_file)]
    """Compares reference and result npy files, returns zero if it passes."""
    assert trc.main(args) == expected

    # Remove files created
    _delete_data_file(reference_file)
    _delete_data_file(result_file)


@pytest.mark.parametrize(
    "data_type,expected",
    [
        (np.int32, trc.TestResult.MISMATCH),
        (np.int64, trc.TestResult.MISMATCH),
        (np.float32, trc.TestResult.MISMATCH),
        (bool, trc.TestResult.MISMATCH),
    ],
)
def test_shape_mismatch(data_type, expected):
    """Check that mismatch shapes do not pass."""
    # Generate and save data as reference and result files to compare.
    npy_data = np.ones(shape=(3, 2), dtype=data_type)
    reference_file = _create_data_file("reference.npy", npy_data)
    npy_data = np.ones(shape=(2, 3), dtype=data_type)
    result_file = _create_data_file("result.npy", npy_data)

    args = [str(reference_file), str(result_file)]
    """Compares reference and result npy files, returns zero if it passes."""
    assert trc.main(args) == expected

    # Remove files created
    _delete_data_file(reference_file)
    _delete_data_file(result_file)


@pytest.mark.parametrize(
    "data_type,expected",
    [
        (np.int32, trc.TestResult.MISMATCH),
        (np.int64, trc.TestResult.MISMATCH),
        (np.float32, trc.TestResult.MISMATCH),
        (bool, trc.TestResult.MISMATCH),
    ],
)
def test_results_mismatch(data_type, expected):
    """Check that different results do not pass."""
    # Generate and save data as reference and result files to compare.
    npy_data = np.zeros(shape=(2, 3), dtype=data_type)
    reference_file = _create_data_file("reference.npy", npy_data)
    npy_data = np.ones(shape=(2, 3), dtype=data_type)
    result_file = _create_data_file("result.npy", npy_data)

    args = [str(reference_file), str(result_file)]
    """Compares reference and result npy files, returns zero if it passes."""
    assert trc.main(args) == expected

    # Remove files created
    _delete_data_file(reference_file)
    _delete_data_file(result_file)


@pytest.mark.parametrize(
    "data_type1,data_type2,expected",
    [  # Pairwise testing of all supported types
        (np.int32, np.int64, trc.TestResult.MISMATCH),
        (bool, np.float32, trc.TestResult.MISMATCH),
    ],
)
def test_types_mismatch(data_type1, data_type2, expected):
    """Check that different types in results do not pass."""
    # Generate and save data as reference and result files to compare.
    npy_data = np.ones(shape=(3, 2), dtype=data_type1)
    reference_file = _create_data_file("reference.npy", npy_data)
    npy_data = np.ones(shape=(3, 2), dtype=data_type2)
    result_file = _create_data_file("result.npy", npy_data)

    args = [str(reference_file), str(result_file)]
    """Compares reference and result npy files, returns zero if it passes."""
    assert trc.main(args) == expected

    # Remove files created
    _delete_data_file(reference_file)
    _delete_data_file(result_file)


@pytest.mark.parametrize(
    "reference_exists,result_exists,expected",
    [
        (True, False, trc.TestResult.MISSING_FILE),
        (False, True, trc.TestResult.MISSING_FILE),
    ],
)
def test_missing_files(reference_exists, result_exists, expected):
    """Check that missing files are caught."""
    # Generate and save data
    npy_data = np.ndarray(shape=(2, 3), dtype=bool)
    reference_file = _create_data_file("reference.npy", npy_data)
    result_file = _create_data_file("result.npy", npy_data)
    if not reference_exists:
        _delete_data_file(reference_file)
    if not result_exists:
        _delete_data_file(result_file)

    args = [str(reference_file), str(result_file)]
    assert trc.main(args) == expected

    if reference_exists:
        _delete_data_file(reference_file)
    if result_exists:
        _delete_data_file(result_file)


@pytest.mark.parametrize(
    "reference_numpy,result_numpy,expected",
    [
        (True, False, trc.TestResult.INCORRECT_FORMAT),
        (False, True, trc.TestResult.INCORRECT_FORMAT),
    ],
)
def test_incorrect_format_files(reference_numpy, result_numpy, expected):
    """Check that incorrect format files are caught."""
    # Generate and save data
    npy_data = np.ndarray(shape=(2, 3), dtype=bool)
    reference_file = (
        _create_data_file("reference.npy", npy_data)
        if reference_numpy
        else _create_empty_file("empty.npy")
    )
    result_file = (
        _create_data_file("result.npy", npy_data)
        if result_numpy
        else _create_empty_file("empty.npy")
    )

    args = [str(reference_file), str(result_file)]
    assert trc.main(args) == expected

    _delete_data_file(reference_file)
    _delete_data_file(result_file)
