# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Validates desc.json against its JSON schema."""
import json
from pathlib import Path

from jsonschema import validate
from referencing import Registry
from referencing import Resource

TD_SCHEMA_FULL = "full"
TD_SCHEMA_DATA_GEN = "data_gen"
TD_SCHEMA_COMPLIANCE = "compliance"

# Default file info
SCRIPT = Path(__file__).absolute()
TD_SCHEMAS = {
    TD_SCHEMA_DATA_GEN: "datagen-config.schema.json",
    TD_SCHEMA_COMPLIANCE: "compliance-config.schema.json",
    TD_SCHEMA_FULL: "desc.schema.json",
}


class TestDescSchemaValidator:
    def __init__(self):
        """Initialize by loading all the schemas and setting up a registry."""
        self.registry = Registry()
        self.schemas = {}
        for key, name in TD_SCHEMAS.items():
            schema_path = SCRIPT.parent / name
            with schema_path.open("r") as fd:
                schema = json.load(fd)
            self.schemas[key] = schema
            if key != TD_SCHEMA_FULL:
                resource = Resource.from_contents(schema)
                self.registry = self.registry.with_resource(uri=name, resource=resource)

    def validate_config(self, config, schema_type=TD_SCHEMA_FULL):
        """Validate the whole (or partial) config versus the relevant schema."""
        validate(config, self.schemas[schema_type], registry=self.registry)


def main(argv=None):
    """Command line interface for the schema validation."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=Path, help="the path to the test directory to validate"
    )
    args = parser.parse_args(argv)
    test_path = args.path

    if not test_path.is_dir():
        print(f"ERROR: Invalid directory - {test_path}")
        return 2

    test_desc_path = test_path / "desc.json"

    if not test_desc_path.is_file():
        print(f"ERROR: No test description found: {test_desc_path}")
        return 2

    # Load the JSON desc.json
    try:
        with test_desc_path.open("r") as fd:
            test_desc = json.load(fd)
    except Exception as e:
        print(f"ERROR: Loading {test_desc_path} - {repr(e)}")
        return 2

    sv = TestDescSchemaValidator()
    sv.validate_config(test_desc, TD_SCHEMA_FULL)

    return 0


if __name__ == "__main__":
    exit(main())
