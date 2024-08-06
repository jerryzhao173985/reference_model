# Conformance

This directory contains the scripts and data files to generate the conformance tests.

The data files are in JSON format and they describe what tests to create and/or select from the `tosa_verif_build_tests` generator script.

## JSON files

### TOSA ops

Naming: `tosa_ext_profile_ops_info.json`

Contains a dictionary of operator names.

Each operator entry contains:

* "group" - name of the group this operator is in, in the spec
* "gen_filter" - optional filter string for op to give to tosa_verif_build_tests - defaults to "^opname$"
* "support_for" - optional list of supported creation modes (for FP tests only) out of:
    * lazy_data_gen - data generation just before test run
    * stable_random_gen - more stable method of generation of tests
    * random_const_inputs - random choice of const or input tensor per op
    * generator_profile_filter - pass the profile and extension
    info to the test builder to use them as filtering
* "generation" - dictionary of test generation details - see below
* "selection" - dictionary of test selection details - see below

In the generation dictionary each entry is a name for a generation group -
a set of tests generated together and then selected from using the selection
criteria.

Each generation group is a dictionary that contains some ways to control when the generation group is run:

* "supports_all" - list of profiles and/or extensions that must all be chosen for generation
* "supports_any" - list of profiles and/or extensions that individually can be chosen for generation
* "from_version" - optional version string for when the tests have been introduced in TOSA
of the form "vM.mm.p" where `M` is the major version, `mm` is the minor version
and `p` is the patch version

Other dictionary entries in a generation group are:

* "no_negative_tests" - optional "true" indicator that no negative tests are relevant/generated
* "negative_dim_range" - optional range of dimensions for negative tests
* "generator_args" - list of argument lists to supply to the `tosa_verif_build_tests` (see that tool for more details)
* "selector" - optional name for the selection criteria to use for this generation group, if not supplied "default" will be used

In the selection dictionary each entry is a name for a selection criteria - there must be one called "default" which is used by default. Others may exist and be used by the different generation groups.

Each selection criteria is a dictionary that contains:

* "all": "true" - to select all tests (and not use test_select)
* "generator_select" - optional "true" to use generator selector instead of conformance test_select

for selection criteria that has "generator_select" set:

* "permutes" - optional list of parameters whose values are to be permuted, the default is ["rank", "dtype"]
* "maximum" - optional number - at most "maximum" tests (not including specific tests) will be captured per permuted "permutes" value, effects "full_params" as well
* "full_params" - optional list of parameter names used to select tests covering a full range of values for these params up to "maximum"
* "specifics" - optional dictionary of params with lists of values, tests that meet any of these "specifics" will be selected and kept (even using "post_sparsity")
* "groups" - optional list of parameters that should be considered as a grouping of tests and treated as one test for "sparsity" and "specifics"
* "num_errorifs" - optional value of error_if tests to keep per error_if case, the default is 1

or for other select criteria it defaults to the old test select (more information for each entry in `test_select.py`):

* "params" - optional dictionary with mappings of parameter names to the values to select
* "permutes" - optional list of parameter names to be permuted
* "preselected" - optional list of dictionaries containing parameter names and pre-chosen values
* "sparsity" - optional dictionary of parameter names with a sparsity value
* "exclude_patterns" - optional list of regex's whereby each match will not be considered for selection. Exclusion happens BEFORE test selection (i.e.
before permutes are applied)
* "errorifs" - optional list of ERRORIF case names to be selected after exclusion (negative tests)

