"""Tests for tosa_reference_model."""
# Copyright (c) 2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from pathlib import Path
from shutil import rmtree

import numpy as np
import pytest
from checker.tosa_result_checker import test_check as tosa_check
from checker.tosa_result_checker import TestResult as TosaResult
from generator.tosa_verif_build_tests import main as tosa_builder
from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError

# Note: Must rename imports so that pytest doesn't assume its a test function/class

# Set this to False if you want ot preserve the test directories after running
CLEAN_UP_TESTS = True

# Location of reference model binary
REF_MODEL_PATH = Path(__file__).resolve().parents[2] / "build" / "reference_model"
REF_MODEL_EXE = "tosa_reference_model"
REF_MODEL = REF_MODEL_PATH / REF_MODEL_EXE

# Default tensor shape information
SHAPE_LIST = ["10", "5"]
SHAPE_DIMS = len(SHAPE_LIST)
SHAPE_ARG = ",".join(SHAPE_LIST)
SHAPE_OUT = "x".join(SHAPE_LIST)

# Output file information
OUTPUT_DIR_PREFIX = "_pytest_vtest"
OUTPUT_OFM_FILE = "result_refmodel_pytest.npy"
OUTPUT_RESULT_FILE = "result_numpy_pytest.npy"
OUTPUT_CONST_GLOB = "const-*.npy"

TEST_DESC_FILENAME = "desc.json"
TOSA_LEVEL = "EIGHTK"

# Conversion from refmodel type into the type abbreviation used in the test output
REF_MODEL_TYPE_TO_OUT = {
    "bool": "b",
    "int8": "i8",
    "uint8": "u8",
    "int16": "i16",
    "int32": "i32",
    "fp32": "f32",
    "fp16": "f16",
    "bf16": "bf16",
}


@pytest.mark.postcommit
def test_refmodel_built():
    """First test to check the reference model has been built."""
    assert REF_MODEL.is_file()


class BuildTosaTest:
    """Wrapper for managing lifecycle of TOSA unit tests."""

    def __init__(self, op_name, ref_model_type, num_expected_tests):
        self.op_name = op_name
        self.ref_model_type = ref_model_type
        self.num_expected_tests = num_expected_tests
        self.output_dir = None
        self.test_dirs = None

    def create_test(self):
        """Helper to generate a TOSA unit test."""
        if self.output_dir is not None:
            # Already created
            return self.test_dir

        self.output_dir = (
            Path(__file__).parent
            / f"{OUTPUT_DIR_PREFIX}_{self.op_name}_{self.ref_model_type}"
        )

        # Generate tests without any zero-point
        build_args = [
            "--filter",
            self.op_name,
            "--target-shape",
            SHAPE_ARG,
            "--target-dtype",
            self.ref_model_type,
            "--zero-point",
            "0",
            "--num-const-inputs-concat",
            "1",
            "--dump-const-tensors",
            "-o",
            str(self.output_dir),
        ]
        print(f"### Building tests: tosa_verif_build_tests {' '.join(build_args)}")
        tosa_builder(build_args)

        # Find the created test
        test_dir = self.output_dir / self.op_name
        # Can't assume exact name due to broadcasting and other changes to shape
        test_glob = f"{self.op_name}_*_{REF_MODEL_TYPE_TO_OUT[self.ref_model_type]}*"
        test_dirs = sorted(test_dir.glob(test_glob))
        assert len(test_dirs) == self.num_expected_tests
        for test_dir in test_dirs:
            assert test_dir.is_dir()
        self.test_dirs = test_dirs

        return self.test_dirs

    def remove_test(self):
        if self.output_dir is not None and self.output_dir.is_dir():
            # Delete directory
            test_tree = self.output_dir.resolve()
            if CLEAN_UP_TESTS:
                print(f"Deleting {test_tree}")
                rmtree(str(test_tree))
                self.output_dir = None
            else:
                print(f"Skipped clean up of {test_tree}")


# Tests - op_name, ref_model_type, num_expected_tests
TEST_PARAMS = [
    ("add", "int32", 1),
    ("add", "fp32", 1),
    ("abs", "int32", 1),
    ("abs", "fp32", 1),
    ("abs", "fp16", 1),
    ("abs", "bf16", 1),
    ("negate", "int8", 1),
    ("negate", "int16", 1),
    ("negate", "int32", 1),
    ("negate", "fp32", 1),
    ("negate", "fp16", 1),
    ("negate", "bf16", 1),
    # One test per axis (shape dimensions)
    ("concat", "bool", SHAPE_DIMS),
    ("concat", "int8", SHAPE_DIMS),
    ("concat", "int16", SHAPE_DIMS),
    ("concat", "int32", SHAPE_DIMS),
    ("concat", "fp32", SHAPE_DIMS),
    ("concat", "fp16", SHAPE_DIMS),
    ("concat", "bf16", SHAPE_DIMS),
]


def id_2_name(id):
    """Convert test id to name - otherwise it will be tosaTestN."""
    op_name, ref_model_type, _ = id
    return f"{op_name}-{ref_model_type}"


@pytest.fixture(params=TEST_PARAMS, ids=id_2_name)
def tosaTest(request):
    """Fixture to generate the required test params and clean up."""
    op_name, ref_model_type, num_expected_tests = request.param
    tst = BuildTosaTest(op_name, ref_model_type, num_expected_tests)
    yield tst
    tst.remove_test()


@pytest.mark.postcommit
def test_refmodel_simple_op(tosaTest):
    """Operator testing versus Numpy."""
    op_name = tosaTest.op_name

    # Generate TOSA test(s) (mostly should be single test)
    test_dirs = tosaTest.create_test()

    # Indicate miscellaneous checks to run in tosa_check
    misc_checks = []

    for test_dir in test_dirs:
        # Run ref model
        desc_file = test_dir / TEST_DESC_FILENAME
        assert desc_file.is_file()
        refmodel_cmd = [
            str(REF_MODEL),
            "--test_desc",
            str(desc_file),
            "--ofm_file",
            OUTPUT_OFM_FILE,
            "--tosa_level",
            TOSA_LEVEL,
        ]
        try:
            run_sh_command(refmodel_cmd, verbose=True, capture_output=True)
        except RunShCommandError as err:
            assert False, f"Unexpected exception {err}"

        # Find output
        ofm_file = test_dir / OUTPUT_OFM_FILE
        assert ofm_file.is_file()

        # Load inputs for Numpy
        with desc_file.open("r") as fp:
            test_desc = json.load(fp)
        tensors = []
        assert "ifm_file" in test_desc
        for input_name in test_desc["ifm_file"]:
            input_file = test_dir / input_name
            assert input_file.is_file()
            tensors.append(np.load(str(input_file)))

        # Load constants for Numpy
        const_files = sorted(test_dir.glob(OUTPUT_CONST_GLOB))
        consts = []
        for const_file in const_files:
            assert const_file.is_file()
            consts.append(np.load(str(const_file)))

        # Perform Numpy operation
        if op_name == "abs":
            assert len(tensors) == 1
            result = np.abs(tensors[0])
        elif op_name == "add":
            assert len(tensors) == 2
            result = np.add(tensors[0], tensors[1])
        elif op_name == "concat":
            assert len(consts) == 1
            # Get axis from test directory name
            match = re.search(r"axis([0-9]+)", test_dir.name)
            assert match is not None
            axis = int(match.group(1))
            result = np.concatenate((*tensors, consts[0]), axis=axis)
        elif op_name == "negate":
            assert len(tensors) == 1
            result = np.negative(tensors[0])
        else:
            assert False, f"Unknown operation {op_name}"

        # Save Numpy result
        result_file = test_dir / OUTPUT_RESULT_FILE
        np.save(str(result_file), result)
        assert result_file.is_file()

        # Ensure valid bf16
        if tosaTest.ref_model_type == "bf16":
            misc_checks.append("bf16")

        # Check Numpy result versus refmodel
        check_result, tolerance, msg = tosa_check(
            result_file,
            ofm_file,
            test_name=test_dir.name,
            misc_checks=misc_checks,
        )
        assert check_result == TosaResult.PASS
