# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Locate model files helper functions."""
from enum import IntEnum
from pathlib import Path

DEFAULT_REF_MODEL_SCHEMA_PATH = Path("thirdparty/serialization_lib/schema")
DEFAULT_REF_MODEL_BUILD_FLATC_PATH = Path(
    "thirdparty/serialization_lib/third_party/flatbuffers"
)
DEFAULT_REF_MODEL_BUILD_EXE_PATH = Path("reference_model")
DEFAULT_BUILD_DIR = Path("build")


class TosaFileType(IntEnum):
    """TOSA file types."""

    REF_MODEL = 0
    SCHEMA = 1
    FLATC = 2
    VERIFY_LIBRARY = 3


TOSA_FILE_TYPE_TO_DETAILS = {
    TosaFileType.REF_MODEL: {
        "name": "tosa_reference_model",
        "location": DEFAULT_REF_MODEL_BUILD_EXE_PATH,
        "build": True,
    },
    TosaFileType.SCHEMA: {
        "name": "tosa.fbs",
        "location": DEFAULT_REF_MODEL_SCHEMA_PATH,
        "build": False,
    },
    TosaFileType.FLATC: {
        "name": "flatc",
        "location": DEFAULT_REF_MODEL_BUILD_FLATC_PATH,
        "build": True,
    },
    TosaFileType.VERIFY_LIBRARY: {
        "name": "libtosa_reference_verify_lib.so",
        "location": DEFAULT_REF_MODEL_BUILD_EXE_PATH,
        "build": True,
    },
}


def find_tosa_file(file_type, ref_model_path, path_is_ref_model_exe=True):
    """Return the possible path to the required tosa file type."""
    name = TOSA_FILE_TYPE_TO_DETAILS[file_type]["name"]
    location = TOSA_FILE_TYPE_TO_DETAILS[file_type]["location"]
    build = TOSA_FILE_TYPE_TO_DETAILS[file_type]["build"]

    if path_is_ref_model_exe:
        # Given a path to the reference_model executable

        # Special case - return what we have been given!
        if file_type == TosaFileType.REF_MODEL:
            return ref_model_path

        try:
            if build:
                # Look in build directory
                search_path = ref_model_path.parents[1]
            else:
                # Look in reference_model directory
                search_path = ref_model_path.parents[2]
        except IndexError:
            search_path = ref_model_path.parent
    else:
        # Given a path to the reference_model directory
        if build:
            search_path = ref_model_path / DEFAULT_BUILD_DIR
        else:
            search_path = ref_model_path

    search_path = search_path / location / name

    return search_path
