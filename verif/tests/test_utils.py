# Copyright (c) 2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from pathlib import Path

import conformance.model_files as cmf
import pytest

# Location of reference model binaries
REF_MODEL_DIR = Path(__file__).resolve().parents[2]
REF_MODEL_EXE_PATH = cmf.find_tosa_file(
    cmf.TosaFileType.REF_MODEL, REF_MODEL_DIR, False
)
VERIFY_EXE_PATH = cmf.find_tosa_file(cmf.TosaFileType.VERIFY, REF_MODEL_EXE_PATH)
GENERATE_LIB_PATH = cmf.find_tosa_file(
    cmf.TosaFileType.GENERATE_LIBRARY, REF_MODEL_EXE_PATH
)
VERIFY_LIB_PATH = cmf.find_tosa_file(
    cmf.TosaFileType.VERIFY_LIBRARY, REF_MODEL_EXE_PATH
)

# Tests for the default locations for model_files

test_params_file_type = [
    cmf.TosaFileType.REF_MODEL,
    cmf.TosaFileType.SCHEMA,
    cmf.TosaFileType.VERIFY_LIBRARY,
    cmf.TosaFileType.GENERATE_LIBRARY,
    cmf.TosaFileType.VERIFY,
]
if sys.platform == "linux":
    # TODO: Enable this test on other platforms
    test_params_file_type.append(cmf.TosaFileType.FLATC)


@pytest.mark.postcommit
@pytest.mark.parametrize("file_type", test_params_file_type)
def test_model_files_filetype(file_type):
    """Test each filetype in model files can be found."""
    file = cmf.find_tosa_file(file_type, REF_MODEL_DIR, False)
    assert file.is_file()

    # Test giving it the full path to the exe
    file_via_exe = cmf.find_tosa_file(file_type, REF_MODEL_EXE_PATH, True)
    assert file == file_via_exe


@pytest.mark.postcommit
def test_model_files_refmodel_default_dir():
    """Test for default ref model location using current working dir."""

    cwd = Path.cwd()
    os.chdir(REF_MODEL_DIR)
    ref_model = cmf.find_tosa_file(cmf.TosaFileType.REF_MODEL, None, False)
    os.chdir(cwd)
    assert ref_model.is_file()
