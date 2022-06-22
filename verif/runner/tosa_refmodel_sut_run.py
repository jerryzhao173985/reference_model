"""TOSA test runner module for the Reference Model."""
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from enum import IntEnum
from enum import unique

from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError
from runner.tosa_test_runner import TosaTestRunner


@unique
class TosaRefReturnCode(IntEnum):
    """Return codes from the Tosa Reference Model."""

    VALID = 0
    UNPREDICTABLE = 1
    ERROR = 2


class TosaSUTRunner(TosaTestRunner):
    """TOSA Reference Model runner."""

    def __init__(self, args, runnerArgs, testDir):
        """Initialize using the given test details."""
        super().__init__(args, runnerArgs, testDir)

    def runTestGraph(self):
        """Run the test on the reference model."""
        # Build up the TOSA reference command line
        # Uses arguments from the argParser args, not the runnerArgs
        args = self.args

        # Call Reference model with description file to provide all file details
        cmd = [
            args.ref_model_path,
            "--operator_fbs={}".format(args.operator_fbs),
            "--test_desc={}".format(self.descFile),
        ]

        # Specific debug options for reference model
        if args.ref_debug:
            cmd.extend(["-d ALL", "-l {}".format(args.ref_debug)])

        if args.ref_intermediates:
            cmd.extend(["-D dump_intermediates=1"])

        # Run command and interpret tosa graph result via process return codes
        graphMessage = None
        try:
            run_sh_command(cmd, self.args.verbose, capture_output=True)
            graphResult = TosaTestRunner.TosaGraphResult.TOSA_VALID
        except RunShCommandError as e:
            graphMessage = e.stderr
            if e.return_code == TosaRefReturnCode.ERROR:
                graphResult = TosaTestRunner.TosaGraphResult.TOSA_ERROR
            elif e.return_code == TosaRefReturnCode.UNPREDICTABLE:
                graphResult = TosaTestRunner.TosaGraphResult.TOSA_UNPREDICTABLE
            else:
                graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR
                if not self.args.verbose:
                    print(e)
        except Exception as e:
            print(e)
            graphMessage = str(e)
            graphResult = TosaTestRunner.TosaGraphResult.OTHER_ERROR

        # Return graph result and message
        return graphResult, graphMessage
