"""Tests for tosa_verif_run_tests.py."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from runner.tosa_verif_run_tests import parseArgs


def test_args_test():
    """Test arguments - test."""
    args = ["-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.test == ["test"]


def test_args_ref_model_path():
    """Test arguments - ref_model_path."""
    args = ["--ref-model-path", "ref_model_path", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.ref_model_path == "ref_model_path"


def test_args_ref_debug():
    """Test arguments - ref_debug."""
    args = ["--ref-debug", "ref_debug", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.ref_debug == "ref_debug"


def test_args_ref_intermediates():
    """Test arguments - ref_intermediates."""
    args = ["--ref-intermediates", "2", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.ref_intermediates == 2


def test_args_verbose():
    """Test arguments - ref_verbose."""
    args = ["-v", "-t", "test"]
    parsed_args = parseArgs(args)
    print(parsed_args.verbose)
    assert parsed_args.verbose == 1


def test_args_jobs():
    """Test arguments - jobs."""
    args = ["-j", "42", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.jobs == 42


def test_args_sut_module():
    """Test arguments - sut_module."""
    args = ["--sut-module", "sut_module", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.sut_module == ["sut_module"]


def test_args_sut_module_args():
    """Test arguments - sut_module_args."""
    args = ["--sut-module-args", "sut_module_args", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.sut_module_args == ["sut_module_args"]


def test_args_xunit_file():
    """Test arguments - xunit-file."""
    args = ["--xunit-file", "xunit_file", "-t", "test"]
    parsed_args = parseArgs(args)
    assert parsed_args.xunit_file == "xunit_file"
