# Copyright (c) 2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import conformance.model_files as cmf

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
