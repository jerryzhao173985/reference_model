"""Tests for tosa_verif_run_tests.py."""
# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from pathlib import Path
from xml.dom import minidom

import pytest
from runner.tosa_test_presets import TOSA_REFCOMPLIANCE_RUNNER
from runner.tosa_verif_run_tests import main


TEST_DESC = {
    "tosa_file": "pytest.json",
    "ifm_name": ["test-0", "test-1"],
    "ifm_file": ["test-0.npy", "test-1.npy"],
    "ofm_name": ["test-result-0"],
    "ofm_file": ["test-result-0.npy"],
    "expected_failure": False,
}
GRAPH_RESULT_VALID = "valid"
GRAPH_RESULT_ERROR = "error"

FAKE_REF_MODEL_PATH = Path(__file__).parent / "__fake_ref_model__"


def _create_fake_ref_model():
    """Create a fake ref model to fool the runner."""
    with FAKE_REF_MODEL_PATH.open("w") as fd:
        print("Fake ref model for mock testing", file=fd)


def _delete_fake_ref_model():
    """Clean up fake ref model."""
    FAKE_REF_MODEL_PATH.unlink()


def _create_desc_json(json_object) -> Path:
    """Create test desc.json."""
    file = Path(__file__).parent / "desc.json"
    with file.open("w") as fd:
        json.dump(json_object, fd, indent=2)
    return file


def _delete_desc_json(file: Path):
    """Clean up desc.json."""
    binary_file = file.parent / "desc_binary.json"
    if binary_file.exists():
        print(binary_file.read_text())
        binary_file.unlink()
    else:
        print(file.read_text())
    file.unlink()


@pytest.fixture
def testDir() -> str:
    """Set up a mock expected pass test."""
    print("SET UP - testDir")
    _create_fake_ref_model()
    file = _create_desc_json(TEST_DESC)
    yield file.parent
    print("TEAR DOWN - testDir")
    _delete_desc_json(file)
    _delete_fake_ref_model()


@pytest.fixture
def testDirExpectedFail() -> str:
    """Set up a mock expected fail test."""
    print("SET UP - testDirExpectedFail")
    _create_fake_ref_model()
    fail = deepcopy(TEST_DESC)
    fail["expected_failure"] = True
    file = _create_desc_json(fail)
    yield file.parent
    print("TEAR DOWN - testDirExpectedFail")
    _delete_desc_json(file)
    _delete_fake_ref_model()


@pytest.fixture
def testDirMultiOutputs() -> str:
    """Set up a mock multiple results output test."""
    print("SET UP - testDirMultiOutputs")
    _create_fake_ref_model()
    out = deepcopy(TEST_DESC)
    out["ofm_name"].append("tr1")
    out["ofm_file"].append("test-result-1.npy")
    file = _create_desc_json(out)
    yield file.parent
    print("TEAR DOWN - testDirMultiOutputs")
    _delete_desc_json(file)
    _delete_fake_ref_model()


def _get_default_argv(testDir: Path, graphResult: str) -> list:
    """Create default args based on test directory and graph result."""
    return [
        "--ref-model-path",
        f"{str(FAKE_REF_MODEL_PATH)}",
        "--sut-module",
        "tests.tosa_mock_sut_run",
        "--test",
        str(testDir),
        "--xunit-file",
        str(testDir / "result.xml"),
        # Must be last argument to allow easy extension with extra args
        "--sut-module-args",
        f"tests.tosa_mock_sut_run:graph={graphResult}",
    ]


def _get_xml_results(argv: list):
    """Get XML results and remove file."""
    resultsFile = Path(argv[argv.index("--xunit-file") + 1])
    results = minidom.parse(str(resultsFile))
    resultsFile.unlink()
    return results


def _get_xml_testsuites_from_results(results, numExpectedTestSuites: int):
    """Get XML testsuites from results."""
    testSuites = results.getElementsByTagName("testsuite")
    assert len(testSuites) == numExpectedTestSuites
    return testSuites


def _check_xml_testsuites_in_results(results, expectedTestSuites: list):
    """Check XML testsuites in results."""
    # Add compliance to expected list
    expectedTestSuites.append(TOSA_REFCOMPLIANCE_RUNNER)
    testSuites = _get_xml_testsuites_from_results(results, len(expectedTestSuites))
    for suite in testSuites:
        assert suite.getAttribute("name") in expectedTestSuites


def _get_xml_testcases_from_results(results, expectedTestCases: int):
    """Get XML testcases from results."""
    testCases = results.getElementsByTagName("testcase")
    assert len(testCases) == expectedTestCases
    return testCases


def _get_xml_failure(argv: list):
    """Get the results and single testcase with the failure result entry if there is one."""
    results = _get_xml_results(argv)
    testCases = _get_xml_testcases_from_results(results, 1)
    fail = testCases[0].getElementsByTagName("failure")
    if fail:
        return fail[0].firstChild.data
    return None


def test_mock_sut_expected_pass(testDir: Path):
    """Run expected pass SUT test."""
    try:
        argv = _get_default_argv(testDir, GRAPH_RESULT_VALID)
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert not fail


UNEXPECTED_PASS_PREFIX_STR = "UNEXPECTED_PASS"
UNEXPECTED_FAIL_PREFIX_STR = "UNEXPECTED_FAIL"


def test_mock_sut_unexpected_pass(testDirExpectedFail: Path):
    """Run unexpected pass SUT test."""
    try:
        argv = _get_default_argv(testDirExpectedFail, GRAPH_RESULT_VALID)
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert fail.startswith(UNEXPECTED_PASS_PREFIX_STR)


def test_mock_sut_expected_failure(testDirExpectedFail: Path):
    """Run expected failure SUT test."""
    try:
        argv = _get_default_argv(testDirExpectedFail, GRAPH_RESULT_ERROR)
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert not fail


def test_mock_sut_unexpected_failure(testDir: Path):
    """Run unexpected failure SUT test."""
    try:
        argv = _get_default_argv(testDir, GRAPH_RESULT_ERROR)
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert fail.startswith(UNEXPECTED_FAIL_PREFIX_STR)


def test_mock_sut_binary_conversion(testDir: Path):
    """Run unexpected failure SUT test."""
    try:
        argv = _get_default_argv(testDir, GRAPH_RESULT_VALID)
        argv.extend(["--binary", "--flatc-path", str(testDir / "mock_flatc.py")])
        main(argv)
        binary_desc = testDir / "desc_binary.json"
        assert binary_desc.exists()
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert not fail


def test_mock_and_dummy_sut_results(testDir: Path):
    """Run two SUTs and check they both return results."""
    try:
        suts = ["tests.tosa_dummy_sut_run", "tests.tosa_mock_sut_run"]
        argv = _get_default_argv(testDir, GRAPH_RESULT_VALID)
        # Override sut-module setting with both SUTs
        argv.extend(["--sut-module"] + suts)
        main(argv)
        results = _get_xml_results(argv)
        _check_xml_testsuites_in_results(results, suts)
        _get_xml_testcases_from_results(results, 2)
    except Exception as e:
        assert False, f"Unexpected exception {e}"


def test_two_mock_suts(testDir: Path):
    """Test that a duplicated SUT is ignored."""
    try:
        sut = ["tests.tosa_mock_sut_run"]
        argv = _get_default_argv(testDir, GRAPH_RESULT_VALID)
        # Override sut-module setting with duplicated SUT
        argv.extend(["--sut-module"] + sut * 2)
        main(argv)
        results = _get_xml_results(argv)
        _check_xml_testsuites_in_results(results, sut)
        _get_xml_testcases_from_results(results, 1)
    except Exception as e:
        assert False, f"Unexpected exception {e}"


def test_mock_sut_multi_outputs_expected_pass(testDirMultiOutputs: Path):
    """Run expected pass SUT test with multiple outputs."""
    try:
        argv = _get_default_argv(testDirMultiOutputs, GRAPH_RESULT_VALID)
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert not fail


def test_mock_sut_multi_outputs_unexpected_failure(testDirMultiOutputs: Path):
    """Run SUT test which expects multiple outputs, but last one is missing."""
    try:
        argv = _get_default_argv(testDirMultiOutputs, GRAPH_RESULT_VALID)
        argv.append("tests.tosa_mock_sut_run:num_results=1")
        main(argv)
        fail = _get_xml_failure(argv)
    except Exception as e:
        assert False, f"Unexpected exception {e}"
    assert fail.startswith(UNEXPECTED_FAIL_PREFIX_STR)
