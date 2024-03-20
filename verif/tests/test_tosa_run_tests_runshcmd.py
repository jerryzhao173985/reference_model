"""Tests for tosa_verif_run_tests.py."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError


def test_run_command_success():
    """Run successful command."""
    cmd = ["echo", "Hello Space Cadets"]
    try:
        run_sh_command(cmd)
        ok = True
    except RunShCommandError:
        ok = False
    assert ok


def test_run_command_fail():
    """Run unsuccessful command."""
    cmd = ["cat", "non-existant-file-432342.txt"]
    try:
        run_sh_command(cmd)
        ok = True
    except RunShCommandError as e:
        assert e.return_code == 1
        ok = False
    assert not ok


def test_run_command_fail_with_stderr():
    """Run unsuccessful command capturing output."""
    cmd = ["cat", "--unknown-option"]
    try:
        stdout, stderr = run_sh_command(cmd, capture_output=True)
        ok = True
    except RunShCommandError as e:
        assert e.return_code == 1
        assert e.stderr
        ok = False
    assert not ok


def test_run_command_success_verbose_with_stdout():
    """Run successful command capturing output."""
    output = "There is no Planet B"
    cmd = ["echo", output]
    try:
        stdout, stderr = run_sh_command(cmd, verbose=True, capture_output=True)
        assert stdout == f"{output}\n"
        ok = True
    except RunShCommandError:
        ok = False
    assert ok
