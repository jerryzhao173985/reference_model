# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Calls the data generation library to create the test data."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np
from schemavalidation import schemavalidation


class GenerateError(Exception):
    """Exception raised for errors performing data generation."""


class GenerateLibrary:
    """Python interface to the C generate library."""

    def __init__(self, generate_lib_path):
        """Find the library and set up the interface."""
        self.lib_path = generate_lib_path
        if not self.lib_path.is_file():
            raise GenerateError(f"Could not find generate library - {self.lib_path}")

        self.test_desc = None
        self.json_config = None
        self.lib = ct.cdll.LoadLibrary(self.lib_path)

        self.tgd_generate_data = self.lib.tgd_generate_data
        self.tgd_generate_data.argtypes = [
            ct.c_char_p,
            ct.c_char_p,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.tgd_generate_data.restype = ct.c_bool

    def check_config(self, test_desc: dict):
        """Quick check that the config supports data generation."""
        return ("meta" in test_desc) and ("data_gen" in test_desc["meta"])

    def set_config(self, test_desc: dict):
        """Set the test config in the library.

        test_desc - the test desc.json file
        """
        self.test_desc = None
        self.json_config = None

        if not self.check_config(test_desc):
            raise GenerateError("No meta/data_gen section found in desc.json")

        # Validate the config versus the schema
        tdsv = schemavalidation.TestDescSchemaValidator()
        tdsv.validate_config(test_desc)

        self.test_desc = test_desc
        self.json_config = test_desc["meta"]["data_gen"]

    def _create_buffer(self, dtype: str, shape: tuple):
        """Helper to create a buffer of the required type."""
        size = 1
        for dim in shape:
            size *= dim

        if dtype == "FP32":
            # Create buffer and initialize to zero
            buffer = (ct.c_float * size)(0)
            size_bytes = size * 4
        else:
            raise GenerateError(f"Unsupported data type {dtype}")

        return buffer, size_bytes

    def _data_gen_write(
        self, test_path: Path, json_bytes: bytes, ifm_name: str, ifm_file: str
    ):
        """Generate the named tensor data and save it in numpy format."""
        try:
            tensor = self.json_config["tensors"][ifm_name]
            dtype = tensor["data_type"]
            shape = tuple(tensor["shape"])
        except KeyError as e:
            raise GenerateError(
                f"Missing data in desc.json for input {ifm_name} - {repr(e)}"
            )

        buffer, size_bytes = self._create_buffer(dtype, shape)
        buffer_ptr = ct.cast(buffer, ct.c_void_p)

        result = self.tgd_generate_data(
            ct.c_char_p(json_bytes),
            ct.c_char_p(bytes(ifm_name, "utf8")),
            buffer_ptr,
            ct.c_size_t(size_bytes),
        )
        if not result:
            raise GenerateError("Data generate failed")

        arr = np.ctypeslib.as_array(buffer)
        arr = np.reshape(arr, shape)

        file_name = test_path / ifm_file
        np.save(file_name, arr)

    def write_numpy_files(self, test_path: Path):
        """Write out all the specified tensors to numpy data files."""
        if self.test_desc is None or self.json_config is None:
            raise GenerateError("Cannot write numpy files as no config set up")

        try:
            ifm_names = self.test_desc["ifm_name"]
            ifm_files = self.test_desc["ifm_file"]
        except KeyError as e:
            raise GenerateError(f"Missing data in desc.json - {repr(e)}")

        json_bytes = bytes(json.dumps(self.json_config), "utf8")

        failures = []
        for iname, ifile in zip(ifm_names, ifm_files):
            try:
                self._data_gen_write(test_path, json_bytes, iname, ifile)
            except GenerateError as e:
                failures.append(
                    f"ERROR: Failed to create data for tensor {iname} - {repr(e)}"
                )

        if len(failures) > 0:
            raise GenerateError("\n".join(failures))


def main(argv=None):
    """Simple command line interface for the data generator."""
    import argparse
    import conformance.model_files as cmf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-lib-path",
        type=Path,
        help="Path to TOSA generate lib",
    )
    parser.add_argument(
        "path", type=Path, help="the path to the test directory to generate data for"
    )
    args = parser.parse_args(argv)
    test_path = args.path

    if args.generate_lib_path is None:
        # Try to work out ref model directory and find the verify library
        # but this default only works for the python developer environment
        # i.e. when using the scripts/py-dev-env.* scripts
        # otherwise use the command line option --generate-lib-path to specify path
        ref_model_dir = Path(__file__).absolute().parents[2]
        args.generate_lib_path = cmf.find_tosa_file(
            cmf.TosaFileType.GENERATE_LIBRARY, ref_model_dir, False
        )

    if not test_path.is_dir():
        print(f"ERROR: Invalid directory - {test_path}")
        return 2

    test_desc_path = test_path / "desc.json"

    if not test_desc_path.is_file():
        print(f"ERROR: No test description found: {test_desc_path}")
        return 2

    # Load the JSON desc.json
    try:
        with test_desc_path.open("r") as fd:
            test_desc = json.load(fd)
    except Exception as e:
        print(f"ERROR: Loading {test_desc_path} - {repr(e)}")
        return 2

    try:
        dgl = GenerateLibrary(args.generate_lib_path)
        if not dgl.check_config(test_desc):
            print(f"WARNING: No data generation supported for {test_path}")
            return 2

        dgl.set_config(test_desc)
    except GenerateError as e:
        print(f"ERROR: Initializing generate library - {repr(e)}")
        return 1

    try:
        dgl.write_numpy_files(test_path)
    except GenerateError as e:
        print(f"ERROR: Writing out data files to {test_path}\n{repr(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
