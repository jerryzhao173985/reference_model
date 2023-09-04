"""Tests for schemavalidation.py."""
# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import pytest
import schemavalidation.schemavalidation as sch
from jsonschema.exceptions import ValidationError


def test_schemavalidation_full_fail():
    json = {}

    sv = sch.TestDescSchemaValidator()
    with pytest.raises(ValidationError) as excinfo:
        sv.validate_config(json)
    info = str(excinfo.value).split("\n")
    assert info[0] == "'tosa_file' is a required property"


def test_schemavalidation_compliance_fail():
    json = {"version": "v"}

    sv = sch.TestDescSchemaValidator()
    with pytest.raises(ValidationError) as excinfo:
        sv.validate_config(json, sch.TD_SCHEMA_COMPLIANCE)
    info = str(excinfo.value).split("\n")
    assert info[0] == "'tensors' is a required property"


def test_schemavalidation_data_gen_fail():
    json = {"version": "v", "tensors": {"input": {}}}

    sv = sch.TestDescSchemaValidator()
    with pytest.raises(ValidationError) as excinfo:
        sv.validate_config(json, sch.TD_SCHEMA_DATA_GEN)
    info = str(excinfo.value).split("\n")
    assert info[0] == "'generator' is a required property"


def test_schemavalidation_full_minimal():
    json = {
        "tosa_file": "file",
        "ifm_name": ["name1", "name2"],
        "ifm_file": ["file1", "file2"],
        "ofm_name": ["name1", "name2"],
        "ofm_file": ["file1", "file2"],
    }

    sv = sch.TestDescSchemaValidator()
    sv.validate_config(json)


def test_schemavalidation_full_unexpected():
    json = {
        "tosa_file": "file",
        "ifm_name": ["name1", "name2"],
        "ifm_file": ["file1", "file2"],
        "ofm_name": ["name1", "name2"],
        "ofm_file": ["file1", "file2"],
        "unexpected_property": 1,
    }

    sv = sch.TestDescSchemaValidator()
    with pytest.raises(ValidationError) as excinfo:
        sv.validate_config(json)
    info = str(excinfo.value).split("\n")
    assert (
        info[0]
        == "Additional properties are not allowed ('unexpected_property' was unexpected)"
    )


def test_schemavalidation_compliance_minimal():
    json = {
        "version": "v",
        "tensors": {
            "output": {
                "mode": "mode",
            }
        },
    }

    sv = sch.TestDescSchemaValidator()
    sv.validate_config(json, sch.TD_SCHEMA_COMPLIANCE)


def test_schemavalidation_data_gen_minimal():
    json = {
        "version": "v",
        "tensors": {
            "input": {
                "generator": "generator",
                "data_type": "type",
                "input_type": "constant",
                "shape": [],
                "op": "name",
                "input_pos": 0,
            }
        },
    }

    sv = sch.TestDescSchemaValidator()
    sv.validate_config(json, sch.TD_SCHEMA_DATA_GEN)
