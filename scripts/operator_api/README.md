# Generate eager operator execution entrypoints

## Introduction

The generate_api.py script will generate an extended reference model API with eager operator execution entrypoints.
The following files will be generated: include/operators.h and src/operators.cc

## Requirements

* Python 3.6 or later
* Jinja2 (install with ```pip install Jinja2```)

## Running from the command line

The script can be run from the scripts/operator-api directory as follows:

```bash
python generate_api.py
```
