import os

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

import json
import shlex
import subprocess
from tosa_test_runner import TosaTestRunner, run_sh_command


class TosaRefRunner(TosaTestRunner):
    def __init__(self, args, runnerArgs, testDir):
        super().__init__(args, runnerArgs, testDir)

    def runModel(self):
        # Build up the TOSA reference command line
        # Uses arguments from the argParser args, not the runnerArgs
        args = self.args

        ref_cmd = [
            args.ref_model_path,
            "-Csubgraph_file={}".format(self.testDesc["tosa_file"]),
            "-Csubgraph_dir={}".format(self.testDir),
            "-Cinput_dir={}".format(self.testDir),
            "-Coutput_dir={}".format(self.testDir),
            "-Coutput_tensor_prefix=ref-",  # Naming agreement with TosaSerializer
        ]

        # Build up input tensor_name/filename list
        inputTensors = []
        for i in range(len(self.testDesc["ifm_placeholder"])):
            inputTensors.append(
                "{}:{}".format(
                    self.testDesc["ifm_placeholder"][i], self.testDesc["ifm_file"][i]
                )
            )

        ref_cmd.append("-Cinput_tensor={}".format(",".join(inputTensors)))

        if args.ref_debug:
            ref_cmd.extend(["-dALL", "-l{}".format(args.ref_debug)])

        if args.ref_intermediates:
            ref_cmd.extend(["-Ddump_intermediates=1"])

        expectedFailure = self.testDesc["expected_failure"]

        try:
            run_sh_command(self.args, ref_cmd)
            if expectedFailure:
                result = TosaTestRunner.Result.UNEXPECTED_PASS
            else:
                result = TosaTestRunner.Result.EXPECTED_PASS
        except Exception as e:
            if expectedFailure:
                result = TosaTestRunner.Result.EXPECTED_FAILURE
            else:
                result = TosaTestRunner.Result.UNEXPECTED_FAILURE

        return result
