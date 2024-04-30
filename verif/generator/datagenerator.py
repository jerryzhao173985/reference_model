# Copyright (c) 2023-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Calls the data generation library to create the test data."""
import ctypes as ct
import json
from pathlib import Path

import numpy as np
import schemavalidation.schemavalidation as sch
from ml_dtypes import bfloat16
from ml_dtypes import float8_e4m3fn
from ml_dtypes import float8_e5m2


class GenerateError(Exception):
    """Exception raised for errors performing data generation."""


class GenerateLibrary:
    """Python interface to the C generate library.

    Simple usage to write out all input files:
      set_config(test_desc)
      write_numpy_files(test_path)

    To get data buffers (for const data):
      get_tensor_data(tensor_name)
    """

    def __init__(self, generate_lib_path):
        """Find the library and set up the interface."""
        self.lib_path = generate_lib_path
        if self.lib_path is None or not self.lib_path.is_file():
            raise GenerateError(f"Could not find generate library - {self.lib_path}")

        self.schema_validator = sch.TestDescSchemaValidator()

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
        self.schema_validator.validate_config(test_desc)

        self.test_desc = test_desc
        self.json_config = test_desc["meta"]["data_gen"]

    def _create_buffer(self, dtype: str, shape: tuple):
        """Helper to create a buffer of the required type."""
        if shape:
            size = np.prod(shape)
        else:
            # Rank 0
            size = 1

        if dtype == "FP32":
            # Create buffer and initialize to zero
            buffer = (ct.c_float * size)(0)
            size_bytes = size * 4
        elif dtype == "FP16" or dtype == "BF16":
            size_bytes = size * 2
            # Create buffer of bytes and initialize to zero
            buffer = (ct.c_ubyte * size_bytes)(0)
        elif dtype == "FP8E4M3" or dtype == "FP8E5M2":
            size_bytes = size
            buffer = (ct.c_ubyte * size_bytes)(0)
        elif dtype == "INT32" or dtype == "SHAPE":
            # Create buffer and initialize to zero
            buffer = (ct.c_int32 * size)(0)
            size_bytes = size * 4
        elif dtype == "INT8":
            size_bytes = size
            # Create buffer of bytes and initialize to zero
            buffer = (ct.c_ubyte * size_bytes)(0)
        else:
            raise GenerateError(f"Unsupported data type {dtype}")

        return buffer, size_bytes

    def _convert_buffer(self, buffer, dtype: str, shape: tuple):
        """Helper to convert a buffer to a numpy array."""
        arr = np.ctypeslib.as_array(buffer)

        if dtype == "FP16":
            # Convert from bytes back to FP16
            arr = np.frombuffer(arr, np.float16)
        elif dtype == "BF16":
            # Convert from bytes back to BF16
            arr = np.frombuffer(arr, bfloat16)
        elif dtype == "FP8E4M3":
            # Convert from bytes back to FP8E4M3
            arr = np.frombuffer(arr, float8_e4m3fn)
        elif dtype == "FP8E5M2":
            # Convert from bytes back to FP8E5M2
            arr = np.frombuffer(arr, float8_e5m2).view(np.uint8)
        arr = np.reshape(arr, shape)

        return arr

    def _data_gen_array(self, json_config: str, tensor_name: str):
        """Generate the named tensor data and return a numpy array."""
        try:
            tensor = json_config["tensors"][tensor_name]
            dtype = tensor["data_type"]
            shape = tuple(tensor["shape"])
        except KeyError as e:
            raise GenerateError(
                f"Missing data in json config for input {tensor_name} - {repr(e)}"
            )

        buffer, size_bytes = self._create_buffer(dtype, shape)
        buffer_ptr = ct.cast(buffer, ct.c_void_p)

        json_bytes = bytes(json.dumps(json_config), "utf8")

        result = self.tgd_generate_data(
            ct.c_char_p(json_bytes),
            ct.c_char_p(bytes(tensor_name, "utf8")),
            buffer_ptr,
            ct.c_size_t(size_bytes),
        )
        if not result:
            raise GenerateError("Data generate failed")

        arr = self._convert_buffer(buffer, dtype, shape)
        return arr

    def _data_gen_write(
        self, test_path: Path, json_config: str, ifm_name: str, ifm_file: str
    ):
        """Generate the named tensor data and save it in numpy format."""
        arr = self._data_gen_array(json_config, ifm_name)

        file_name = test_path / ifm_file
        np.save(file_name, arr)

    def write_numpy_files(self, test_path: Path):
        """Write out all the desc.json input tensors to numpy data files."""
        if self.test_desc is None or self.json_config is None:
            raise GenerateError("Cannot write numpy files as no config set up")

        try:
            ifm_names = self.test_desc["ifm_name"]
            ifm_files = self.test_desc["ifm_file"]
        except KeyError as e:
            raise GenerateError(f"Missing data in desc.json - {repr(e)}")

        failures = []
        for iname, ifile in zip(ifm_names, ifm_files):
            try:
                self._data_gen_write(test_path, self.json_config, iname, ifile)
            except GenerateError as e:
                failures.append(
                    f"ERROR: Failed to create data for tensor {iname} - {repr(e)}"
                )

        if len(failures) > 0:
            raise GenerateError("\n".join(failures))

    def get_tensor_data(self, tensor_name: str, json_config=None):
        """Get a numpy array for a named tensor in the data_gen meta data."""
        if json_config is None:
            if self.json_config is None:
                raise GenerateError("Cannot get tensor data as no config set up")
            json_config = self.json_config
        else:
            # Validate the given config
            self.schema_validator.validate_config(
                json_config, schema_type=sch.TD_SCHEMA_DATA_GEN
            )

        return self._data_gen_array(json_config, tensor_name)


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
