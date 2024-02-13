# Conformance

This directory contains the scripts and data files to generate the conformance tests.

The data files are in JSON format and they describe what tests to create and/or select from the `tosa_verif_build_tests` and `tosa_verif_framework_*` generator scripts.

## JSON files

### TOSA ops

Naming: `tosa_PPP_profile_ops_info.json`

Contains a dictionary of operator names.
Where `PPP` is the profile subset of either `base` for all integer tests, or `main` for all floating point tests.

Each operator entry contains:

* "group" - name of the group this operator is in, in the spec
* "profile" - list of profiles that this operator covers
* "gen_filter" - optional filter string for op to give to tosa_verif_build_tests - defaults to "^opname$"
* "support_for" - optional list of supported creation modes out of:
    * lazy_data_gen - data generation just before test run
    * generator_select - use generator selector instead of conformance test_select
* "generation" - dictionary of test generation details - see below
* "selection" - dictionary of test selection details - see below

In the generation dictionary each entry is a name for a generation group -
a set of tests generated together and then selected from using the selection
criteria.

Each generation group is a dictionary that contains:

* "from_version" - optional version string for when the tests have been introduced in TOSA
of the form "vM.mm.p" where `M` is the major version, `mm` is the minor version
and `p` is the patch version
* "no_negative_tests" - optional "true" indicator that no negative tests are relevant/generated
* "negative_dim_range" - optional range of dimensions for negative tests
* "generator_args" - list of argument lists to supply to the `tosa_verif_build_tests` (see that tool for more details)
* "selector" - optional name for the selection criteria to use for this generation group, if not supplied "default" will be used

In the selection dictionary each entry is a name for a selection criteria - there must be one called "default" which is used by default. Others may exist and be used by the different generation groups.

Each selection criteria is a dictionary that contains:

* "all": "true" - to select all tests (and not use test_select)

or for operators that have "support_for" "generator_select":

* "permutes" - optional list of parameters whose values are to be permuted, the default is ["rank", "dtype"]
* "maximum" - optional number - at most "maximum" tests (not including specific tests) will be captured per permuted "permutes" value, effects "full_params" as well
* "full_params" - optional list of parameter names used to select tests covering a full range of values for these params up to "maximum"
* "specifics" - optional dictionary of params with lists of values, tests that meet any of these "specifics" will be selected and kept (even using "post_sparsity")
* "groups" - optional list of parameters that should be considered as a grouping of tests and treated as one test for "sparsity" and "specifics"
* "num_errorifs" - optional value of error_if tests to keep per error_if case, the default is 1

or for other operators it defaults to the old test select (more information for each entry in `test_select.py`):

* "params" - optional dictionary with mappings of parameter names to the values to select
* "permutes" - optional list of parameter names to be permuted
* "preselected" - optional list of dictionaries containing parameter names and pre-chosen values
* "sparsity" - optional dictionary of parameter names with a sparsity value
* "exclude_patterns" - optional list of regex's whereby each match will not be considered for selection. Exclusion happens BEFORE test selection (i.e.
before permutes are applied)
* "errorifs" - optional list of ERRORIF case names to be selected after exclusion (negative tests)

### Framework ops

DEPRECATED - not supported for conformance testing.

NOTE: Currently assumed all framework ops will be TFLite.

Naming: `tosa_PPP_profile_framework_ops_info.json`

Contains a dictionary of operator names.
Where `PPP` is the profile subset of either `base` for all integer tests, or `main` for all floating point tests.

Each operator entry contains:

* "tests" - list of tests to be part of conformance
* "profile" - list of profiles that these tests cover
* "alternate_names" - optional list of names that are used by the framework test generator and can be renamed to the operator name.

Example:
```
    "average_pool_2d": {
        "alternate_names": [
            "avg_pool2d"
        ],
        "tests": [
            "average_pool_2d_1x4x4x4_qi8_st11_padSAME_kern11",
            "average_pool_2d_1x4x8x19_qi16_st21_padSAME_kern22",
            "average_pool_2d_1x7x7x9_qi8_st22_padSAME_kern11",
            "average_pool_2d_1x32x32x8_qu8_st12_padVALID_kern12",
            "average_pool_2d_1x8x4x17_qu8_st21_padVALID_kern21"
        ],
        "profile": [
            "tosa-bi",
            "tosa-mi"
        ]
    },
```