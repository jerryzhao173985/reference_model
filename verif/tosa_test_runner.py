import os

# Copyright (c) 2020, ARM Limited.
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
from enum import IntEnum, unique


def run_sh_command(args, full_cmd, capture_output=False):
    """Utility function to run an external command. Optionally return captured stdout/stderr"""

    # Quote the command line for printing
    full_cmd_esc = [shlex.quote(x) for x in full_cmd]

    if args.verbose:
        print("### Running {}".format(" ".join(full_cmd_esc)))

    if capture_output:
        rc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if rc.returncode != 0:
            print(rc.stdout.decode("utf-8"))
            print(rc.stderr.decode("utf-8"))
            raise Exception(
                "Error running command: {}.\n{}".format(
                    " ".join(full_cmd_esc), rc.stderr.decode("utf-8")
                )
            )
        return (rc.stdout, rc.stderr)
    else:
        rc = subprocess.run(full_cmd)
    if rc.returncode != 0:
        raise Exception("Error running command: {}".format(" ".join(full_cmd_esc)))


class TosaTestRunner:
    def __init__(self, args, runnerArgs, testDir):

        self.args = args
        self.runnerArgs = runnerArgs
        self.testDir = testDir

        # Load the json test file
        with open(os.path.join(testDir, "desc.json"), "r") as fd:
            self.testDesc = json.load(fd)

    def runModel(self):
        pass

    class Result(IntEnum):
        EXPECTED_PASS = 0
        EXPECTED_FAILURE = 1
        UNEXPECTED_PASS = 2
        UNEXPECTED_FAILURE = 3
        INTERNAL_ERROR = 4
