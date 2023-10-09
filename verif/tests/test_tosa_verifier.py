"""Tests for the python interface to the verifier library."""
# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest
from checker.verifier import VerifierError
from checker.verifier import VerifierLibrary

# NOTE: These tests are marked as POST COMMIT
# To run them, please build the reference_model in a local "build" directory
# (as per the README) and run them using: pytest -m "postcommit"

# Location of reference model binaries
REF_MODEL_BUILD_PATH = Path(__file__).resolve().parents[2] / "build" / "reference_model"
VERIFIER_LIB = "libtosa_reference_verify_lib.so"
VERIFIER_LIB_PATH = REF_MODEL_BUILD_PATH / VERIFIER_LIB


@pytest.mark.postcommit
def test_verifier_lib_built():
    """First test to check the library has been built."""
    assert VERIFIER_LIB_PATH.is_file()


@pytest.mark.postcommit
def test_checker_verifier_load_fail():
    with pytest.raises(VerifierError) as excinfo:
        VerifierLibrary(Path("/place-that-does-not-exist"))
    assert str(excinfo.value).startswith("Could not find verify library")


@pytest.mark.postcommit
def test_checker_verifier_load():
    vlib = VerifierLibrary(VERIFIER_LIB_PATH)
    assert vlib


JSON_COMPLIANCE_DOT_PRODUCT = {
    "version": "0.1",
    "tensors": {
        "output1": {
            "mode": "DOT_PRODUCT",
            "data_type": "FP32",
            "dot_product_info": {"ks": 1000, "s": 0},
        }
    },
}


@pytest.mark.postcommit
def test_checker_verifier_dot_product_check():
    vlib = VerifierLibrary(VERIFIER_LIB_PATH)
    assert vlib

    imp_arr = np.zeros((10, 10, 10), dtype=np.float32)
    ref_arr = np.zeros((10, 10, 10), dtype=np.float64)
    bnd_arr = np.zeros((10, 10, 10), dtype=np.float64)

    json_config = JSON_COMPLIANCE_DOT_PRODUCT

    ret = vlib.verify_data("output1", json_config, imp_arr, ref_arr, bnd_arr)
    assert ret


@pytest.mark.postcommit
def test_checker_verifier_dot_product_check_fail():
    vlib = VerifierLibrary(VERIFIER_LIB_PATH)
    assert vlib

    imp_arr = np.zeros((10, 10, 10), dtype=np.float32)
    ref_arr = np.ones((10, 10, 10), dtype=np.float64)
    bnd_arr = np.zeros((10, 10, 10), dtype=np.float64)

    json_config = JSON_COMPLIANCE_DOT_PRODUCT

    ret = vlib.verify_data("output1", json_config, imp_arr, ref_arr, bnd_arr)
    assert not ret
