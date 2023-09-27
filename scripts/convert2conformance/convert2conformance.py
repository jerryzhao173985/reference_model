#!/usr/bin/env python3
# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""This script converts generated tests into conformance tests.

It can convert a framework unit test or a reference model unit test.
It expects the tests have been already run on the reference model
so it can capture the result as the expected result.
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from json2fbbin.json2fbbin import fbbin_to_json
from json2numpy.json2numpy import npy_to_json
from schemavalidation.schemavalidation import TestDescSchemaValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert2conformance")


NAME_FLATBUFFER_DIR = ["flatbuffer-", "_FW_"]
NAME_DESC_FILENAME = "desc.json"
NAME_CONFORMANCE_RESULT_PREFIX = "Conformance-"
NAME_REFMODEL_RUN_RESULT_SUFFIX = ".runner.tosa_refmodel_sut_run.npy"

PROFILES_LIST = ["tosa-bi", "tosa-mi"]


def parse_args(argv):
    """Parse the arguments."""
    # Set prog for when we are called via tosa_verif_conformance_generator
    parser = argparse.ArgumentParser(prog="convert2conformance")
    parser.add_argument(
        "test_dir",
        default=Path.cwd(),
        type=Path,
        nargs="?",
        help="The test directory to convert (default is CWD)",
    )
    parser.add_argument(
        "--schema-path",
        "--operator-fbs",
        dest="schema_path",
        type=Path,
        required=True,
        help=("Path to reference model schema."),
    )
    parser.add_argument(
        "--flatc-path",
        dest="flatc_path",
        type=Path,
        required=True,
        help=("Path to flatc executable."),
    )
    parser.add_argument(
        "--output-directory",
        dest="output_dir",
        type=Path,
        default=Path.cwd() / "conformance",
        help="Output directory (default is conformance in CWD)",
    )
    parser.add_argument(
        "--framework",
        dest="framework",
        choices=["tflite"],
        default="tflite",
        help="Framework to convert (default tflite)",
    )
    parser.add_argument(
        "--framework-schema",
        dest="framework_schema",
        type=Path,
        help="Framework schema needed to convert framework models",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        choices=PROFILES_LIST,
        action="append",
        required=True,
        help="Profiles this test is suitable for. May be repeated",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        type=str,
        help="Optional string tag to mark this test with. May be repeated",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Output directory must not contain the same test directory",
    )
    parser.add_argument(
        "--lazy-data-generation",
        action="store_true",
        help="Enable lazy data generation (only for tosa-mi)",
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Verbose operation"
    )
    args = parser.parse_args(argv)

    return args


def find_framework_artifacts(framework: str, schema_path: Path, desc_file: Path):
    """Check that any required schema has been supplied for conversion."""
    if framework == "tflite":
        if not schema_path:
            raise Exception("the following arguments are required: --framework-schema")
        elif not schema_path.is_file():
            raise Exception(f"framework schema not found at {schema_path}")
        model = desc_file.parent.parent / "model.tflite"
        if not model.is_file():
            raise Exception(f"Model file not found at {model}")
        return schema_path, model
    return None, None


def get_framework_name(name_array: list, framework: str):
    """Get the framework conversion directory name."""
    name = ""
    for part in name_array:
        if part == "_FW_":
            part = framework
        name = f"{name}{part}"
    return name


def convert_flatbuffer_file(flatc: Path, schema: Path, model_file: Path, output: Path):
    """Convert the flatbuffer binary into JSON."""
    try:
        fbbin_to_json(flatc, schema, model_file, output)
    except Exception as e:
        logger.error(f"Failed to convert flatbuffer binary:\n{e}")
        return None

    if model_file.name == "model.tflite":
        file_name = "model-tflite.json"
        os.rename(output / "model.json", output / file_name)
    else:
        file_name = model_file.stem + ".json"
    return output / file_name


def convert_numpy_file(n_file: Path, output: Path, outname: Optional[str] = None):
    """Convert a numpy file into a JSON file."""
    j_file = output / (outname if outname else (n_file.stem + ".json"))
    npy_to_json(n_file, j_file)
    return j_file


def update_desc_json(
    test_dir: Path,
    test_desc,
    output_dir: Optional[Path] = None,
    create_result=True,
    profiles=None,
    tags=None,
):
    """Update the desc.json format for conformance and optionally create result."""
    ofm_files = []
    cfm_files = []
    if not output_dir:
        output_dir = test_dir
    for index, ofm in enumerate(test_desc["ofm_file"]):
        ofm_path = test_dir / ofm
        if not test_desc["expected_failure"]:
            cfm = NAME_CONFORMANCE_RESULT_PREFIX + test_desc["ofm_name"][index]
            if create_result:
                if ofm_path.is_file():
                    # Use the desc.json name
                    ofm_refmodel = ofm_path
                else:
                    # Adjust for renaming due to tosa_verif_run_tests
                    ofm_refmodel = ofm_path.with_suffix(NAME_REFMODEL_RUN_RESULT_SUFFIX)
                # Create conformance result
                if ofm_refmodel.is_file():
                    convert_numpy_file(ofm_refmodel, output_dir, outname=cfm + ".json")
                else:
                    logger.error(f"Missing result file {ofm_path}")
                    return None
                cfm_files.append(cfm + ".npy")
        # Remove path and "ref-"/"ref_model_" from output filenames
        ofm_files.append(strip_ref_output_name(ofm_path.name))

    # Rewrite output file names as they can be relative, but keep them npys
    test_desc["ofm_file"] = ofm_files
    if not test_desc["expected_failure"] and cfm_files:
        # Output expected result file for conformance if expected pass and we
        # have some files!
        test_desc["expected_result_file"] = cfm_files

    # Add supported profiles
    if profiles is None:
        # Assume base profile
        profiles = [PROFILES_LIST[0]]
    test_desc["profile"] = profiles

    # Add tags (if any)
    if tags is not None:
        test_desc["tag"] = tags

    return test_desc


def strip_ref_output_name(name):
    """Remove mentions of reference from output files."""
    if name.startswith("ref-"):
        name = name[4:]
    if name.startswith("ref_model_"):
        name = name[10:]
    return name


def main(argv=None):
    """Convert the given directory to a conformance test."""
    args = parse_args(argv)
    # Verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check we can get the files we need
    if not args.flatc_path.is_file():
        logger.error("flatc not found at %s", args.flatc_path)
        return 2
    if not args.schema_path.is_file():
        logger.error("TOSA schema not found at %s", args.schema_path)
        return 2

    # Work out where the desc.json file is
    desc_filename = args.test_dir / NAME_DESC_FILENAME
    framework_conversion = False
    test_type_desc = "unknown"
    if desc_filename.is_file():
        logger.debug("Found TOSA operator unit test")
        test_type_desc = "TOSA operator"
    else:
        desc_filename = (
            args.test_dir
            / get_framework_name(NAME_FLATBUFFER_DIR, args.framework)
            / NAME_DESC_FILENAME
        )
        if desc_filename.is_file():
            logger.debug(f"Found framework unit test for {args.framework}")
            test_type_desc = f"{args.framework}"
            framework_conversion = True
        else:
            logger.error(f"Could not find {NAME_DESC_FILENAME} in {args.test_dir}")
            return 2
    logger.debug(f"desc.json file: {desc_filename}")

    # Check for required files for framework conversion
    if framework_conversion:
        try:
            framework_schema, framework_filename = find_framework_artifacts(
                args.framework, args.framework_schema, desc_filename
            )
        except Exception as err:
            logger.error(err)
            return 2
    else:
        framework_schema, framework_filename = None, None

    # Open the meta desc.json file
    with open(desc_filename, mode="r") as fd:
        test_desc = json.load(fd)

    if "tosa_file" not in test_desc:
        logger.error(f"Unsupported desc.json file found {desc_filename}")
        return 2

    # Dictionary fix
    if "ifm_name" not in test_desc:
        logger.warn("Old format desc.json file found - attempting to fix up")
        test_desc["ifm_name"] = test_desc["ifm_placeholder"]
        del test_desc["ifm_placeholder"]

    # Make the output directory if needed
    try:
        args.output_dir.mkdir(parents=True, exist_ok=(not args.strict))
    except FileExistsError:
        if args.strict:
            logger.error(f"{args.output_dir} already exists")
        else:
            logger.error(f"{args.output_dir} is not a directory")
        return 2

    # Convert the TOSA flatbuffer binary
    tosa_filename = desc_filename.parent / test_desc["tosa_file"]
    tosa_filename = convert_flatbuffer_file(
        args.flatc_path, args.schema_path, tosa_filename, args.output_dir
    )
    if not tosa_filename:
        # Failed to convert the file, json2fbbin will have printed an error
        return 1
    else:
        # Replace binary with JSON name
        test_desc["tosa_file"] = tosa_filename.name

    if framework_conversion and framework_filename:
        # Convert the framework flatbuffer binary
        framework_filename = convert_flatbuffer_file(
            args.flatc_path, framework_schema, framework_filename, args.output_dir
        )
        if not framework_filename:
            # Failed to convert the file, json2fbbin will have printed an error
            return 1

    # Convert input files to JSON
    ifm_files = []
    for file in test_desc["ifm_file"]:
        if file:
            path = desc_filename.parent / file
            ifm_files.append(path.name)
            if path.is_file():
                convert_numpy_file(path, args.output_dir)
            else:
                if not args.lazy_data_generation:
                    logger.error(f"Missing input file {path.name}")
                    return 1

    # Rewrite input file names to make sure the paths are correct,
    # but keep them numpys as the test runner will convert them back
    # before giving them to the SUT
    test_desc["ifm_file"] = ifm_files

    # Check for cpp files for data-generator/verifier
    cpp_files = args.test_dir.glob("*.cpp")
    for cpp in cpp_files:
        shutil.copy(str(cpp), str(args.output_dir))

    # Update desc.json and convert result files to JSON
    test_desc = update_desc_json(
        desc_filename.parent,
        test_desc,
        output_dir=args.output_dir,
        create_result=(not args.lazy_data_generation),
        profiles=args.profile,
        tags=args.tags,
    )
    if not test_desc:
        # Error from conversion/update
        return 1

    # Validate the desc.json schema
    try:
        TestDescSchemaValidator().validate_config(test_desc)
    except Exception as e:
        logger.error(e)
        return 1

    # Output new desc.json
    new_desc_filename = args.output_dir / NAME_DESC_FILENAME
    with open(new_desc_filename, "w") as fd:
        json.dump(test_desc, fd, indent=2)

    logger.info(f"Converted {test_type_desc} test to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
