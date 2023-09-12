# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Verfier library interface."""
import ctypes as ct
import json
from pathlib import Path
from typing import Optional

import numpy as np
import schemavalidation.schemavalidation as sch


# Default library info
SCRIPT = Path(__file__).absolute()
# NOTE: This REFMODEL_DIR default only works for the python developer environment
# i.e. when using the scripts/py-dev-env.* scripts
# otherwise use the command line option --ref-model-directory to specify path
REFMODEL_DIR = SCRIPT.parents[2]
LIBRARY = "libtosa_reference_verify_lib.so"

# Type conversion from numpy to tosa_datatype_t
# "type" matches enum - see include/types.h
# "size" is size in bytes per value of this datatype
NUMPY_DATATYPE_TO_CLIENTTYPE = {
    # tosa_datatype_int32_t (all integer types are this!)
    np.dtype("int32"): {"type": 5, "size": 4},
    # tosa_datatype_int48_t (or SHAPE)
    np.dtype("int64"): {"type": 6, "size": 8},
    # tosa_datatype_fp16_t
    np.dtype("float16"): {"type": 2, "size": 2},
    # tosa_datatype_fp32_t (bf16 stored as this)
    np.dtype("float32"): {"type": 3, "size": 4},
    # tosa_datatype_fp64_t (for precise refmodel data)
    np.dtype("float64"): {"type": 99, "size": 8},
    # tosa_datatype_bool_t
    np.dtype("bool"): {"type": 1, "size": 1},
}


class TosaTensor(ct.Structure):
    _fields_ = [
        ("name", ct.c_char_p),
        ("shape", ct.POINTER(ct.c_int32)),
        ("num_dims", ct.c_int32),
        ("data_type", ct.c_int),
        ("data", ct.POINTER(ct.c_uint8)),
        ("size", ct.c_size_t),
    ]


class VerifierError(Exception):
    """Exception raised for errors performing data generation."""


class VerifierLibrary:
    """Python interface to the C verify library."""

    def __init__(self, path: Optional[Path] = None):
        """Find the library and set up the interface."""
        if path is None:
            path = REFMODEL_DIR
        lib_paths = sorted(path.glob(f"**/{LIBRARY}"))

        if len(lib_paths) < 1:
            raise VerifierError(
                f"Could not find {LIBRARY} - have you built the ref-model?"
            )

        self.lib_path = lib_paths[0]
        self.lib = ct.cdll.LoadLibrary(self.lib_path)

        self.tvf_verify_data = self.lib.tvf_verify_data
        self.tvf_verify_data.argtypes = [
            ct.POINTER(TosaTensor),  # ref
            ct.POINTER(TosaTensor),  # ref_bnd
            ct.POINTER(TosaTensor),  # imp
            ct.c_char_p,  # config_json
        ]
        self.tvf_verify_data.restype = ct.c_bool

    def _get_tensor_data(self, name, array):
        """Set up tosa_tensor_t using the given a numpy array."""
        shape = (ct.c_int32 * len(array.shape))(*array.shape)
        size_in_bytes = array.size * NUMPY_DATATYPE_TO_CLIENTTYPE[array.dtype]["size"]

        tensor = TosaTensor(
            ct.c_char_p(bytes(name, "utf8")),
            ct.cast(shape, ct.POINTER(ct.c_int32)),
            ct.c_int32(len(array.shape)),
            ct.c_int(NUMPY_DATATYPE_TO_CLIENTTYPE[array.dtype]["type"]),
            ct.cast(np.ctypeslib.as_ctypes(array), ct.POINTER(ct.c_uint8)),
            ct.c_size_t(size_in_bytes),
        )
        return tensor

    def verify_data(
        self,
        output_name,
        compliance_json_config,
        imp_result_array,
        ref_result_array,
        bnd_result_array=None,
    ):
        """Verify the data using the verification library."""
        sch.TestDescSchemaValidator().validate_config(
            compliance_json_config, sch.TD_SCHEMA_COMPLIANCE
        )
        jsb = bytes(json.dumps(compliance_json_config), "utf8")

        imp = self._get_tensor_data(output_name, imp_result_array)
        ref = self._get_tensor_data(output_name, ref_result_array)
        if bnd_result_array is not None:
            ref_bnd = self._get_tensor_data(output_name, bnd_result_array)
        else:
            ref_bnd = None

        result = self.tvf_verify_data(ref, ref_bnd, imp, ct.c_char_p(jsb))

        return result


def main(argv=None):
    """Simple command line interface for the verifier library."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-model-directory",
        dest="ref_model_dir",
        default=REFMODEL_DIR,
        type=Path,
        help="Path to pre-built reference model directory",
    )
    parser.add_argument(
        "--test-desc",
        type=Path,
        help="Path to test description file: desc.json",
    )
    parser.add_argument(
        "-n",
        "--ofm-name",
        dest="ofm_name",
        type=str,
        help="output tensor name to check (defaults to only ofm_name in desc.json)",
    )
    parser.add_argument(
        "--bnd-result-path",
        type=Path,
        help="path to the reference bounds result numpy file",
    )

    parser.add_argument(
        "ref_result_path", type=Path, help="path to the reference result numpy file"
    )
    parser.add_argument(
        "imp_result_path",
        type=Path,
        help="path to the implementation result numpy file",
    )
    args = parser.parse_args(argv)

    if args.test_desc:
        json_path = args.test_desc
    else:
        # Assume its with the reference file
        json_path = args.ref_result_path.parent / "desc.json"

    print("Load test description")
    with json_path.open("r") as fd:
        test_desc = json.load(fd)

    if args.ofm_name is None:
        if len(test_desc["ofm_name"]) != 1:
            print("ERROR: ambiguous output to check, please specify output tensor name")
            return 2
        output_name = test_desc["ofm_name"][0]
    else:
        output_name = args.ofm_name

    if "meta" not in test_desc or "compliance" not in test_desc["meta"]:
        print(f"ERROR: no compliance meta-data found in {str(json_path)}")
        return 2

    print("Load numpy data")
    paths = [args.imp_result_path, args.ref_result_path, args.bnd_result_path]
    arrays = [None, None, None]
    for idx, path in enumerate(paths):
        if path is not None:
            array = np.load(path)
        else:
            array = None
        arrays[idx] = array

    print("Load verifier library")
    vlib = VerifierLibrary(args.ref_model_dir)

    print("Verify data")
    if vlib.verify_data(output_name, test_desc["meta"]["compliance"], *arrays):
        print("SUCCESS")
        return 0
    else:
        print("FAILURE - NOT COMPLIANT")
        return 1


if __name__ == "__main__":
    exit(main())
