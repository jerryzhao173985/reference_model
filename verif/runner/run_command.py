"""Shell command runner function."""
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import shlex
import subprocess


class RunShCommandError(Exception):
    """Exception raised for errors running the shell command.

    Attributes:
        return_code - non-zero return code from running command
        full_cmd_esc - command and arguments list (pre-escaped)
        stderr - (optional) - standard error output
    """

    def __init__(self, return_code, full_cmd_esc, stderr=None, stdout=None):
        """Initialize run shell command error."""
        self.return_code = return_code
        self.full_cmd_esc = full_cmd_esc
        self.stderr = stderr
        self.stdout = stdout
        self.message = "Error {} running command: {}".format(
            self.return_code, " ".join(self.full_cmd_esc)
        )
        if stdout:
            self.message = "{}\n{}".format(self.message, self.stdout)
        if stderr:
            self.message = "{}\n{}".format(self.message, self.stderr)
        super().__init__(self.message)


def run_sh_command(full_cmd, verbose=False, capture_output=False):
    """Run an external shell command.

    full_cmd: array containing shell command and its arguments
    verbose: optional flag that enables verbose output
    capture_output: optional flag to return captured stdout/stderr
    """
    # Quote the command line for printing
    full_cmd_esc = [shlex.quote(x) for x in full_cmd]

    if verbose:
        print("### Running {}".format(" ".join(full_cmd_esc)))

    if capture_output:
        rc = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = rc.stdout.decode("utf-8")
        stderr = rc.stderr.decode("utf-8")
        if verbose:
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="")
    else:
        stdout, stderr = None, None
        rc = subprocess.run(full_cmd)

    if rc.returncode != 0:
        raise RunShCommandError(rc.returncode, full_cmd_esc, stderr, stdout)
    return (stdout, stderr)
