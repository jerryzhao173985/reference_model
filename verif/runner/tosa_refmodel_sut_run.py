"""TOSA test runner module for the Reference Model."""
# Copyright (c) 2020-2023, ARM Limited.
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

    def __init__(self, args, runnerArgs, testDirPath):
        """Initialize using the given test details."""
        super().__init__(args, runnerArgs, testDirPath)

        # Don't do any compliance runs
        self.compliance = False

    def runTestGraph(self):
        """Run the test on the reference model."""
        # Build up the TOSA reference command line
        # Uses arguments from the argParser args, not the runnerArgs
        args = self.args

        # Call Reference model with description file to provide all file details
        cmd = [
            str(args.ref_model_path),
            f"--tosa_level={args.tosa_level}",
            f"--operator_fbs={str(args.schema_path)}",
            f"--test_desc={self.descFile}",
        ]

        # Specific debug options for reference model
        if args.ref_debug:
            cmd.extend(["-d", "ALL", "-l", args.ref_debug])

        if args.ref_intermediates:
            cmd.extend(["--dump_intermediates", str(args.ref_intermediates)])

        if args.precise_mode or self.compliance:
            cmd.extend(["--precise_mode=1"])

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
