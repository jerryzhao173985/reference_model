"""Template test runner class for running TOSA tests."""
# Copyright (c) 2020-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
from enum import IntEnum

import schemavalidation.schemavalidation as sch
from checker.color_print import LogColors
from checker.color_print import print_color
from checker.color_print import set_print_in_color
from checker.tosa_result_checker import set_print_result
from checker.tosa_result_checker import test_check
from generator.datagenerator import GenerateLibrary
from json2fbbin import json2fbbin
from json2numpy import json2numpy
from runner.tosa_test_presets import TOSA_REFCOMPLIANCE_RUNNER


def isComplianceAbsModeNeeded(testDesc):
    """Checks the test descriptor for DOT_PRODUCT/ABS_ERROR compliance mode."""
    if (
        "meta" in testDesc
        and "compliance" in testDesc["meta"]
        and "tensors" in testDesc["meta"]["compliance"]
    ):
        for _, t in testDesc["meta"]["compliance"]["tensors"].items():
            if "mode" in t and t["mode"] in ("DOT_PRODUCT", "ABS_ERROR"):
                return True
        return False


def getRunnerResultFilePath(resultFilePath, sutModule):
    """Return the result file path with the runner specific naming."""
    return resultFilePath.with_suffix(f".{sutModule}{resultFilePath.suffix}")


def getBoundsResultFilePath(resultFilePath, sutModule=None):
    """Return the bounds result file with/without runner specific naming."""
    boundsFilePath = resultFilePath.parent / f"bounds_{resultFilePath.name}"
    if sutModule is not None:
        boundsFilePath = boundsFilePath.with_suffix(
            f".{sutModule}{boundsFilePath.suffix}"
        )
    return boundsFilePath


class TosaTestInvalid(Exception):
    """Exception raised for errors loading test description.

    Attributes:
        path - full path to missing test description file
        exception = underlying exception
    """

    def __init__(self, path, exception):
        """Initialize test not found error."""
        self.path = path
        self.exception = exception
        self.message = "Invalid test, could not read test description {}: {}".format(
            self.path, str(self.exception)
        )
        super().__init__(self.message)


class TosaTestRunner:
    """TOSA Test Runner template class for systems under test."""

    def __init__(self, args, runnerArgs, testDirPath):
        """Initialize and load JSON meta data file."""
        self.args = args
        self.runnerArgs = runnerArgs
        self.testDir = str(testDirPath)
        self.testDirPath = testDirPath
        self.testName = self.testDirPath.name
        self.verify_lib_path = args.verify_lib_path
        self.generate_lib_path = args.generate_lib_path

        set_print_in_color(not args.no_color)
        # Stop the result checker printing anything - we will do it
        set_print_result(False)

        # Check if we want to run binary and if its already converted
        descFilePath = testDirPath / "desc.json"
        descBinFilePath = testDirPath / "desc_binary.json"
        if args.binary:
            if descBinFilePath.is_file():
                descFilePath = descBinFilePath

        try:
            # Load the json test file
            with descFilePath.open("r") as fd:
                self.testDesc = json.load(fd)
            # Validate the json with the schema
            sch.TestDescSchemaValidator().validate_config(self.testDesc)
        except Exception as e:
            raise TosaTestInvalid(str(descFilePath), e)

        # Convert to binary if needed
        tosaFilePath = testDirPath / self.testDesc["tosa_file"]
        if args.binary and tosaFilePath.suffix == ".json":
            # Convert tosa JSON to binary
            json2fbbin.json_to_fbbin(
                args.flatc_path,
                args.schema_path,
                tosaFilePath,
                testDirPath,
            )
            # Write new desc_binary file
            self.testDesc["tosa_file"] = tosaFilePath.stem + ".tosa"
            with descBinFilePath.open("w") as fd:
                json.dump(self.testDesc, fd, indent=2)
            descFilePath = descBinFilePath

        # Set location of desc.json (or desc_binary.json) file in use
        self.descFile = str(descFilePath)
        self.descFilePath = descFilePath

        # Check for compliance mode - need to run refmodel to get results
        if "meta" in self.testDesc and "compliance" in self.testDesc["meta"]:
            self.complianceMode = True
            if "expected_result" in self.testDesc:
                if self.args.verbose:
                    print("Warning: fixing conflicting compliance mode in test.desc")
                self.testDesc.pop("expected_result")
        else:
            self.complianceMode = False

    def skipTest(self):
        """Check if the test is skipped due to test type or profile selection."""
        expectedFailure = self.testDesc["expected_failure"]
        if self.args.test_type == "negative" and not expectedFailure:
            return True, "non-negative type"
        elif self.args.test_type == "positive" and expectedFailure:
            return True, "non-positive type"
        if self.args.profile:
            profile = self.testDesc["profile"] if "profile" in self.testDesc else []
            if self.args.profile not in profile:
                return True, "non-{} profile".format(self.args.profile)
        return False, ""

    def _ready_file(self, dataFile, jsonOnly=False):
        """Convert/create any data file that is missing."""
        dataPath = self.testDirPath / dataFile
        if not dataPath.is_file():
            jsonPath = dataPath.with_suffix(".json")
            if jsonPath.is_file():
                # Data files stored as JSON
                if self.args.verbose:
                    print(f"Readying data file: {dataPath}")
                json2numpy.json_to_npy(jsonPath)
            elif not jsonOnly:
                # Use data generator for all data files
                if self.args.verbose:
                    print("Readying all data input files")
                dgl = GenerateLibrary(self.generate_lib_path)
                dgl.set_config(self.testDesc)
                dgl.write_numpy_files(self.testDirPath)

    def readyDataFiles(self):
        """Check that the data files have been created/converted."""
        for dataFile in self.testDesc["ifm_file"]:
            self._ready_file(dataFile)
        # Convert expected result if any
        if "expected_result_file" in self.testDesc:
            for dataFile in self.testDesc["expected_result_file"]:
                self._ready_file(dataFile, jsonOnly=True)

    def runTestGraph(self):
        """Override with function that calls system under test."""
        pass

    def testResult(self, tosaGraphResult, graphMessage=None):
        """Work out test result based on graph result and output files."""
        expectedFailure = self.testDesc["expected_failure"]
        print_check_result = False

        sutModule = self.__module__

        if tosaGraphResult == TosaTestRunner.TosaGraphResult.TOSA_VALID:
            if expectedFailure:
                result = TosaTestRunner.Result.UNEXPECTED_PASS
                resultMessage = "Expected failure test incorrectly passed"
            else:
                # Work through all the results produced by the testing, assuming success
                # but overriding this with any failures found
                result = TosaTestRunner.Result.EXPECTED_PASS
                messages = []

                # Go through each output result checking it
                for resultNum, resultFileName in enumerate(self.testDesc["ofm_file"]):
                    resultFilePath = self.testDirPath / resultFileName

                    # Work out the file to check against (if any)
                    if self.complianceMode and sutModule != TOSA_REFCOMPLIANCE_RUNNER:
                        conformanceFilePath = getRunnerResultFilePath(
                            resultFilePath, TOSA_REFCOMPLIANCE_RUNNER
                        )
                        if isComplianceAbsModeNeeded(self.testDesc):
                            conformanceBoundsPath = getBoundsResultFilePath(
                                resultFilePath, TOSA_REFCOMPLIANCE_RUNNER
                            )
                        else:
                            # Not expecting a bounds file for this test
                            conformanceBoundsPath = None
                    elif "expected_result_file" in self.testDesc:
                        conformanceBoundsPath = None
                        try:
                            conformanceFilePath = (
                                self.testDirPath
                                / self.testDesc["expected_result_file"][resultNum]
                            )
                        except IndexError:
                            result = TosaTestRunner.Result.INTERNAL_ERROR
                            msg = "Internal error: Missing expected_result_file {} in {}".format(
                                resultNum, self.descFile
                            )
                            messages.append(msg)
                            print(msg)
                            break
                    else:
                        # Nothing to check against
                        conformanceFilePath = None
                        conformanceBoundsPath = None

                    if conformanceFilePath:
                        print_check_result = True  # Result from checker
                        chkResult, tolerance, msg = test_check(
                            conformanceFilePath,
                            resultFilePath,
                            test_name=self.testName,
                            test_desc=self.testDesc,
                            bnd_result_path=conformanceBoundsPath,
                            ofm_name=self.testDesc["ofm_name"][resultNum],
                            verify_lib_path=self.verify_lib_path,
                        )
                        # Change EXPECTED_PASS assumption if we have any failures
                        if chkResult != 0:
                            result = TosaTestRunner.Result.UNEXPECTED_FAILURE
                            messages.append(msg)
                            if self.args.verbose:
                                print(msg)
                    else:
                        # No conformance file to verify, just check results file exists
                        if not resultFilePath.is_file():
                            result = TosaTestRunner.Result.UNEXPECTED_FAILURE
                            msg = f"Results file is missing: {resultFilePath}"
                            messages.append(msg)
                            print(msg)

                    if resultFilePath.is_file():
                        # Move the resultFilePath to allow subsequent system under
                        # tests to create them and to test they have been created
                        # and to enable compliance testing against refmodel results
                        resultFilePath.rename(
                            getRunnerResultFilePath(resultFilePath, sutModule)
                        )
                        if (
                            isComplianceAbsModeNeeded(self.testDesc)
                            and sutModule == TOSA_REFCOMPLIANCE_RUNNER
                        ):
                            boundsFilePath = getBoundsResultFilePath(resultFilePath)
                            if boundsFilePath.is_file():
                                boundsFilePath = boundsFilePath.rename(
                                    getBoundsResultFilePath(resultFilePath, sutModule)
                                )
                            else:
                                result = TosaTestRunner.Result.INTERNAL_ERROR
                                msg = f"Internal error: Missing expected dot product compliance bounds file {boundsFilePath}"
                                messages.append(msg)
                                print(msg)

                resultMessage = "\n".join(messages) if len(messages) > 0 else None
        else:
            if (
                expectedFailure
                and tosaGraphResult == TosaTestRunner.TosaGraphResult.TOSA_ERROR
            ):
                result = TosaTestRunner.Result.EXPECTED_FAILURE
                resultMessage = None
            else:
                result = TosaTestRunner.Result.UNEXPECTED_FAILURE
                resultMessage = graphMessage

        status = "Result" if print_check_result else "Result code"
        if (
            result == TosaTestRunner.Result.EXPECTED_FAILURE
            or result == TosaTestRunner.Result.EXPECTED_PASS
        ):
            print_color(LogColors.GREEN, f"{sutModule}: {status} PASS {self.testName}")
        else:
            print_color(LogColors.RED, f"{sutModule}: {status} FAIL {self.testName}")

        return result, resultMessage

    class Result(IntEnum):
        """Test result codes."""

        EXPECTED_PASS = 0
        EXPECTED_FAILURE = 1
        UNEXPECTED_PASS = 2
        UNEXPECTED_FAILURE = 3
        INTERNAL_ERROR = 4
        SKIPPED = 5

    class TosaGraphResult(IntEnum):
        """The tosa_graph_result codes."""

        TOSA_VALID = 0
        TOSA_UNPREDICTABLE = 1
        TOSA_ERROR = 2
        OTHER_ERROR = 3
