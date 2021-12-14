# Copyright (c) 2021-2022 Arm Limited.
# SPDX-License-Identifier: Apache-2.0

# Source this script to install the python tosa tools as editable:
#       . scripts/py-dev-env.sh

# NOTE: This script is needed to fix up PYTHONPATH due to a bug
# in setuptools that does not support multiple package_dirs
# properly: https://github.com/pypa/setuptools/issues/230

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please source this within a python virtual env"
    return
fi

if [ -e "setup.cfg" ]; then
    pip install -e .
    export PYTHONPATH=$PWD/scripts:$PWD/thirdparty/serialization_lib/python
    echo "Set PYTHONPATH=$PYTHONPATH"
else
    echo "Please source this from the root of reference_model"
fi
