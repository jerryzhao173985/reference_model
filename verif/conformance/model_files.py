# Copyright (c) 2023-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Locate model files helper functions."""
import sys
from enum import IntEnum
from pathlib import Path

DEFAULT_REF_MODEL_SCHEMA_PATH = Path("thirdparty/serialization_lib/schema")
# This is the Linux/Mac location for FLATC built in SerLib
DEFAULT_FLATC_PATH = Path("thirdparty/serialization_lib/third_party/flatbuffers")
FLATC_IN_BUILD = False

DEFAULT_REF_MODEL_BUILD_EXE_PATH = Path("reference_model")
DEFAULT_VERIFY_BUILD_EXE_PATH = Path("reference_model/verify")
DEFAULT_BUILD_DIR = Path("build")
BUILD_SUB_DIR = ""
LIBRARY_PREFIX = "lib"
EXE_SUFFIX = ""

if sys.platform == "linux":
    LIBRARY_SUFFIX = "so"
elif sys.platform == "darwin":
    LIBRARY_SUFFIX = "dylib"
elif sys.platform == "win32":
    BUILD_SUB_DIR = "Release"
    LIBRARY_PREFIX = ""
    LIBRARY_SUFFIX = "dll"
    EXE_SUFFIX = ".exe"
    DEFAULT_FLATC_PATH = Path("_deps/flatbuffers-build")
    FLATC_IN_BUILD = True


class TosaFileType(IntEnum):
    """TOSA file types."""

    REF_MODEL = 0
    SCHEMA = 1
    FLATC = 2
    VERIFY_LIBRARY = 3
    GENERATE_LIBRARY = 4
    VERIFY = 5


TOSA_FILE_TYPE_TO_DETAILS = {
    TosaFileType.REF_MODEL: {
        "name": f"tosa_reference_model{EXE_SUFFIX}",
        "location": DEFAULT_REF_MODEL_BUILD_EXE_PATH,
        "build": True,
    },
    TosaFileType.SCHEMA: {
        "name": "tosa.fbs",
        "location": DEFAULT_REF_MODEL_SCHEMA_PATH,
        "build": False,
    },
    TosaFileType.FLATC: {
        "name": f"flatc{EXE_SUFFIX}",
        "location": DEFAULT_FLATC_PATH,
        "build": FLATC_IN_BUILD,
    },
    TosaFileType.VERIFY_LIBRARY: {
        "name": f"{LIBRARY_PREFIX}tosa_reference_verify_lib.{LIBRARY_SUFFIX}",
        "location": DEFAULT_REF_MODEL_BUILD_EXE_PATH,
        "build": True,
    },
    TosaFileType.GENERATE_LIBRARY: {
        "name": f"{LIBRARY_PREFIX}tosa_reference_generate_lib.{LIBRARY_SUFFIX}",
        "location": DEFAULT_REF_MODEL_BUILD_EXE_PATH,
        "build": True,
    },
    TosaFileType.VERIFY: {
        "name": f"tosa_verify{EXE_SUFFIX}",
        "location": DEFAULT_VERIFY_BUILD_EXE_PATH,
        "build": True,
    },
}


def get_default_location(file_type, build_path=DEFAULT_BUILD_DIR):
    """Return the default location in the reference_model folder."""
    name = TOSA_FILE_TYPE_TO_DETAILS[file_type]["name"]
    location = TOSA_FILE_TYPE_TO_DETAILS[file_type]["location"]
    build = TOSA_FILE_TYPE_TO_DETAILS[file_type]["build"]

    if build:
        return build_path / location / BUILD_SUB_DIR / name
    else:
        return location / name


def find_tosa_file(file_type, ref_model_path, path_is_ref_model_exe=True):
    """Return the possible path to the required tosa file type."""
    build = TOSA_FILE_TYPE_TO_DETAILS[file_type]["build"]

    if ref_model_path is None:
        # Assume current directory is the reference_model
        ref_model_path = Path.cwd()
        path_is_ref_model_exe = False

    if path_is_ref_model_exe:
        # Given a path to the reference_model executable

        # Special case - return what we have been given!
        if file_type == TosaFileType.REF_MODEL:
            return ref_model_path

        try:
            extra_level = 1 if BUILD_SUB_DIR else 0
            if build:
                # Look in build directory
                search_path = ref_model_path.parents[1 + extra_level]
            else:
                # Look in reference_model directory
                search_path = ref_model_path.parents[2 + extra_level]
        except IndexError:
            search_path = ref_model_path.parent
    else:
        # Given a path to the reference_model directory
        if build:
            search_path = ref_model_path / DEFAULT_BUILD_DIR
        else:
            search_path = ref_model_path

    # Already added the build folder in previous stage
    search_path = search_path / get_default_location(file_type, build_path="")

    return search_path
