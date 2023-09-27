"""Template test runner class for running TOSA tests."""
# Copyright (c) 2020-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
from enum import IntEnum

from checker.tosa_result_checker import LogColors
from checker.tosa_result_checker import print_color
from checker.tosa_result_checker import set_print_in_color
from checker.tosa_result_checker import test_check
from json2fbbin import json2fbbin


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

        set_print_in_color(not args.no_color)

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

    def runTestGraph(self):
        """Override with function that calls system under test."""
        pass

    def testResult(self, tosaGraphResult, graphMessage=None):
        """Work out test result based on graph result and output files."""
        expectedFailure = self.testDesc["expected_failure"]
        print_result_line = True

        if tosaGraphResult == TosaTestRunner.TosaGraphResult.TOSA_VALID:
            if expectedFailure:
                result = TosaTestRunner.Result.UNEXPECTED_PASS
                resultMessage = "Expected failure test incorrectly passed"
            else:
                # Work through all the results produced by the testing, assuming success
                # but overriding this with any failures found
                result = TosaTestRunner.Result.EXPECTED_PASS
                messages = []
                for resultNum, resultFileName in enumerate(self.testDesc["ofm_file"]):
                    if "expected_result_file" in self.testDesc:
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
                        conformanceFilePath = None
                    resultFilePath = self.testDirPath / resultFileName

                    if conformanceFilePath:
                        print_result_line = False  # Checker will print one for us
                        chkResult, tolerance, msg = test_check(
                            conformanceFilePath,
                            resultFilePath,
                            test_name=self.testName,
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
                            msg = "Results file is missing: {}".format(resultFilePath)
                            messages.append(msg)
                            print(msg)

                    if resultFilePath.is_file():
                        # Move the resultFilePath to allow subsequent system under
                        # tests to create them and to test they have been created
                        resultFilePath = resultFilePath.rename(
                            resultFilePath.with_suffix(
                                ".{}{}".format(self.__module__, resultFilePath.suffix)
                            )
                        )

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

        if print_result_line:
            if (
                result == TosaTestRunner.Result.EXPECTED_FAILURE
                or result == TosaTestRunner.Result.EXPECTED_PASS
            ):
                print_color(
                    LogColors.GREEN, "Result code PASS {}".format(self.testName)
                )
            else:
                print_color(LogColors.RED, "Result code FAIL {}".format(self.testName))

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
