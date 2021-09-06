# Copyright (c) 2020-2021, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
import shlex
import subprocess
from enum import Enum, IntEnum, unique
from tosa_test_runner import TosaTestRunner, run_sh_command


@unique
class TosaReturnCode(IntEnum):
    VALID = 0
    UNPREDICTABLE = 1
    ERROR = 2


class TosaRefRunner(TosaTestRunner):
    def __init__(self, args, runnerArgs, testDir):
        super().__init__(args, runnerArgs, testDir)

    def runModel(self):
        # Build up the TOSA reference command line
        # Uses arguments from the argParser args, not the runnerArgs
        args = self.args

        ref_cmd = [
            args.ref_model_path,
            "-Ctest_desc={}".format(os.path.join(self.testDir, "desc.json")),
        ]

        if args.ref_debug:
            ref_cmd.extend(["-dALL", "-l{}".format(args.ref_debug)])

        if args.ref_intermediates:
            ref_cmd.extend(["-Ddump_intermediates=1"])

        expectedReturnCode = self.testDesc["expected_return_code"]

        try:
            rc = run_sh_command(self.args, ref_cmd)
            if rc == TosaReturnCode.VALID:
                if expectedReturnCode == TosaReturnCode.VALID:
                    result = TosaTestRunner.Result.EXPECTED_PASS
                else:
                    result = TosaTestRunner.Result.UNEXPECTED_PASS
            elif rc == TosaReturnCode.ERROR:
                if expectedReturnCode == TosaReturnCode.ERROR:
                    result = TosaTestRunner.Result.EXPECTED_FAILURE
                else:
                    result = TosaTestRunner.Result.UNEXPECTED_FAILURE
            elif rc == TosaReturnCode.UNPREDICTABLE:
                if expectedReturnCode == TosaReturnCode.UNPREDICTABLE:
                    result = TosaTestRunner.Result.EXPECTED_FAILURE
                else:
                    result = TosaTestRunner.Result.UNEXPECTED_FAILURE
            else:
                raise Exception(f"Return code ({rc}) unknown.")

        except Exception as e:
            raise Exception("Runtime Error when running: {}".format(" ".join(ref_cmd)))

        return result
