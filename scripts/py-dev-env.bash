# Copyright (c) 2021-2022 Arm Limited.
# SPDX-License-Identifier: Apache-2.0

# Source this script to install the python tosa tools as editable:
#       . scripts/py-dev-env.bash

# NOTE: This script is needed to fix up PYTHONPATH due to a bug
# in setuptools that does not support multiple package_dirs
# properly: https://github.com/pypa/setuptools/issues/230

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please source this within a python virtual env"
    return
fi

if [ -e "${BASH_SOURCE[0]}" ]; then
    SCRIPTS_DIR=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
    REFMODEL_DIR=$(dirname "$SCRIPTS_DIR")
    pip install -e "$REFMODEL_DIR"
    export PYTHONPATH="$SCRIPTS_DIR:$REFMODEL_DIR/thirdparty/serialization_lib/python"
    echo "Set PYTHONPATH=$PYTHONPATH"
else
    echo "Please source this using bash"
fi
