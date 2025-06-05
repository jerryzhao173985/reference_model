"""Color printing module."""
# Copyright (c) 2020-2023,2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from enum import Enum
from enum import unique

if sys.platform == "win32":
    # Enable ANSI color printing on Windows
    os.system("color")

color_printing = True


@unique
class LogColors(Enum):
    """Shell escape sequence colors for logging."""

    NONE = "\u001b[0m"
    GREEN = "\u001b[32;1m"
    RED = "\u001b[31;1m"
    YELLOW = "\u001b[33;1m"
    BOLD_WHITE = "\u001b[1m"


def set_print_in_color(enabled):
    """Set color printing to enabled or disabled."""
    global color_printing
    color_printing = enabled


def print_color(color, msg):
    """Print color status messages if enabled."""
    global color_printing
    if not color_printing:
        print(msg)
    else:
        print("{}{}{}".format(color.value, msg, LogColors.NONE.value))
