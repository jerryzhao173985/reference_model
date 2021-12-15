"""TOSA test runner module for a mock System Under Test (SUT)."""
# Copyright (c) 2021, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import os

from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError
from runner.tosa_test_runner import TosaTestRunner


class TosaSUTRunner(TosaTestRunner):
    """TOSA mock SUT runner."""

    def __init__(self, args, runnerArgs, testDir):
        """Initialize using the given test details."""
        super().__init__(args, runnerArgs, testDir)

    def runTestGraph(self):
        """Run the test on a mock SUT."""
        # Read the command line sut-module-args in form arg=value
        # and put them in a dictionary
        # Note: On the command line (for this module) they look like:
        #       tests.tosa_mock_sut_run:arg=value
        sutArgs = {}
        for runArg in self.runnerArgs:
            try:
                arg, value = runArg.split("=", 1)
            except ValueError:
                # Argument without a value - treat it as a flag
                arg = runArg
                value = True
            sutArgs[arg] = value
        print(f"MOCK SUT: Runner argument dictionary: {sutArgs}")

        # Useful meta data and arguments
        tosaFlatbufferSchema = self.args.operator_fbs
        tosaSubgraphFile = self.testDesc["tosa_file"]
        tosaTestDirectory = self.testDir
        tosaTestDescFile = self.descFile

        # Expected file name for the graph results on valid graph
        graphResultFiles = []
        for idx, name in enumerate(self.testDesc["ofm_name"]):
            graphResultFiles.append(
                "{}:{}".format(name, self.testDesc["ofm_file"][idx])
            )

        # Build up input "tensor_name":"filename" list
        tosaInputTensors = []
        for idx, name in enumerate(self.testDesc["ifm_name"]):
            tosaInputTensors.append(
                "{}:{}".format(name, self.testDesc["ifm_file"][idx])
            )

        # Build up command line
        cmd = [
            "echo",
            f"FBS={tosaFlatbufferSchema}",
            f"Path={tosaTestDirectory}",
            f"Desc={tosaTestDescFile}",
            f"Graph={tosaSubgraphFile}",
            "Results={}".format(",".join(graphResultFiles)),
            "Inputs={}".format(",".join(tosaInputTensors)),
        ]

        # Run test on implementation
        graphResult = None
        graphMessage = None
        try:
            stdout, stderr = run_sh_command(cmd, verbose=True, capture_output=True)
        except RunShCommandError as e:
            # Return codes can be used to indicate graphResult status (see tosa_ref_run.py)
            # But in this mock version we just set the result based on sutArgs below
            print(f"MOCK SUT: Unexpected error {e.return_code} from command: {e}")
            graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR
            graphMessage = e.stderr

        # Other mock system testing
        if self.args.binary:
            # Check that the mock binary conversion has happened
            _, ext = os.path.splitext(tosaSubgraphFile)
            if (
                os.path.basename(tosaTestDescFile) != "desc_binary.json"
                and ext != ".tosa"
            ):
                graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR

        # Mock up graph result based on passed arguments
        if not graphResult:
            try:
                if sutArgs["graph"] == "valid":
                    graphResult = TosaTestRunner.TosaGraphResult.TOSA_VALID
                    # Create dummy output file(s) for passing result checker
                    for idx, fname in enumerate(self.testDesc["ofm_file"]):
                        if "num_results" in sutArgs and idx == int(
                            sutArgs["num_results"]
                        ):
                            # Skip writing any more to test results checker
                            break
                        print("Created " + fname)
                        fp = open(os.path.join(tosaTestDirectory, fname), "w")
                        fp.close()
                elif sutArgs["graph"] == "error":
                    graphResult = TosaTestRunner.TosaGraphResult.TOSA_ERROR
                    graphMessage = "MOCK SUT: ERROR_IF"
                elif sutArgs["graph"] == "unpredictable":
                    graphResult = TosaTestRunner.TosaGraphResult.TOSA_UNPREDICTABLE
                    graphMessage = "MOCK SUT: UNPREDICTABLE"
                else:
                    graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR
                    graphMessage = "MOCK SUT: error from system under test"
            except KeyError:
                graphMessage = "MOCK SUT: No graph result specified!"
                print(graphMessage)
                graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR

        # Return graph result and message
        return graphResult, graphMessage
