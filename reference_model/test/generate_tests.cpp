// Copyright (c) 2023-2024, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include "generate.h"
#include "half.hpp"

#include <doctest.h>

#include <array>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void update_json_template(std::string& str, const std::string& find, const std::string& change)
{
    // Update the 'str' by looking for instances of 'find' and replacing them with 'change'
    auto pos = str.find(find);
    while (pos != std::string::npos)
    {
        str.replace(pos, find.length(), change);
        pos = str.find(find);
    }
}

template <typename T>
void check_value(bool match, T result, T expected, uint32_t idx)
{
    std::stringstream msg;
    msg << "index: " << idx << " expected: " << std::hex << expected << " got: " << result;
    if (match)
    {
        REQUIRE_MESSAGE(expected == result, msg.str());
    }
    else
    {
        REQUIRE_MESSAGE(expected != result, msg.str());
    }
}

template <typename T>
void check_output(const std::vector<T>& results, const std::vector<uint32_t>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        check_value(true, *(uint32_t*)&results[idx], expected[idx], idx);
    }
}

template <typename T>
void check_output(const std::vector<T>& results, const std::vector<uint16_t>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        check_value(true, *(uint16_t*)&results[idx], expected[idx], idx);
    }
}

template <typename T>
void check_output(const std::vector<T>& results, const std::vector<T>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        check_value(true, *(uint32_t*)&results[idx], *(uint32_t*)&expected[idx], idx);
    }
}

template <typename T>
void check_not_output(const std::vector<T>& results, const std::vector<T>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        check_value(false, *(uint32_t*)&results[idx], *(uint32_t*)&expected[idx], idx);
    }
}

}    // namespace

TEST_SUITE_BEGIN("generate");

TEST_CASE("negative - api")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "in1" : {
                "generator": "_GENERATOR_",
                "data_type": "_TYPE_",
                "input_type": "VARIABLE",
                "shape" : [ 4, 8, 8 ],
                "input_pos": 0,
                "op" : "_OP_",
                "dot_product_info": {
                    "s": 0,
                    "ks": 8,
                    "acc_type": "_TYPE_"
                }
            }
        }
    })";

    const std::string tosaName = "in1";
    const size_t tosaElements  = 4 * 8 * 8;
    const size_t tosaSize      = tosaElements * 4;

    SUBCASE("missing input")
    {
        REQUIRE_FALSE(tgd_generate_data(NULL, NULL, NULL, 0));
    }
    SUBCASE("invalid json")
    {
        std::string invalidJsonCfg = R"({
            "tensors" : {
                "in1" : {
                    "generator": DOT_PRODUCT,
                },
            }
        })";

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(invalidJsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("unknown generator")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_GENERATOR_", "SOLAR");
        update_json_template(jsonCfg, "_TYPE_", "FP32");
        update_json_template(jsonCfg, "_OP_", "MATMUL");
        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("unknown op")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_GENERATOR_", "DOT_PRODUCT");
        update_json_template(jsonCfg, "_TYPE_", "FP32");
        update_json_template(jsonCfg, "_OP_", "GREEN");

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("unknown type")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_GENERATOR_", "DOT_PRODUCT");
        update_json_template(jsonCfg, "_TYPE_", "WATT");
        update_json_template(jsonCfg, "_OP_", "MATMUL");

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("mismatching name")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_GENERATOR_", "DOT_PRODUCT");
        update_json_template(jsonCfg, "_TYPE_", "FP32");
        update_json_template(jsonCfg, "_OP_", "MATMUL");
        std::string invalidName = "notFound1";

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(jsonCfg.c_str(), invalidName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("mismatching size")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_GENERATOR_", "DOT_PRODUCT");
        update_json_template(jsonCfg, "_TYPE_", "FP32");
        update_json_template(jsonCfg, "_OP_", "MATMUL");
        size_t smallElements = 4 * 8 * 7;
        size_t smallSize     = smallElements * 4;

        std::vector<float> buffer(smallElements);
        REQUIRE_FALSE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), smallSize));
    }
}

void matmul_test_FP32(const std::string tosaName[2],
                      const size_t tosaElements[2],
                      const std::string templateJsonCfg,
                      const std::string setStr,
                      int32_t param,
                      const std::vector<uint32_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);
    std::vector<float> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 4));
    check_output<float>(buffer, expected);
}

TEST_CASE("positive - FP32 matmul dot product (first 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "in1" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 4, 8, 2 ],
                "input_pos": 0,
                "op" : "MATMUL",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 2,
                    "acc_type": "FP32"
                }
            },
            "in2" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 4, 2, 5 ],
                "input_pos": 1,
                "op" : "MATMUL",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 2,
                    "acc_type": "FP32"
                }
            }

        }
    })";

    const std::string tosaName[2] = { "in1", "in2" };
    const size_t tosaElements[2]  = { (4 * 8 * 2), (4 * 2 * 5) };

    SUBCASE("matmul, set 0, param 0")
    {
        std::vector<uint32_t> expected = { 0xbf665aa4, 0xbf736bd3, 0x0 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 0, expected);
    }
    SUBCASE("matmul, set 0, param 1")
    {
        std::vector<uint32_t> expected = { 0x0, 0x0, 0x3f34f2dd };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 1, expected);
    }
    SUBCASE("matmul, set 1, param 0")
    {
        std::vector<uint32_t> expected = { 0x5e97f1b0, 0x5ea6a18e, 0x5eb811af };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("matmul, set 1, param 1")
    {
        std::vector<uint32_t> expected = { 0x5f128bb1, 0x5ef54579, 0x5ebd65b8 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("matmul, set 2, param 0")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3f14567c, 0x3f800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, expected);
    }
    SUBCASE("matmul, set 2, param 1")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3f800000, 0x3f800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, expected);
    }
    SUBCASE("matmul, set 3, param 0")
    {
        std::vector<uint32_t> expected = { 0x41800000, 0x3fee6533, 0x41800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, expected);
    }
    SUBCASE("matmul, set 3, param 1")
    {
        std::vector<uint32_t> expected = { 0xc1800000, 0x41800000, 0xc1800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, expected);
    }
    SUBCASE("matmul, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0xbf000000, 0x0 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, expected);
    }
    SUBCASE("matmul, set 4, param 1")
    {
        std::vector<uint32_t> expected = { 0x5dee53e9, 0xdf0cfb23, 0x5f06cb46 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 1, expected);
    }
    SUBCASE("matmul, set 5, param 0")
    {
        std::vector<uint32_t> expected = { 0x5df6c4b3, 0x5e6b4088, 0x5ed0fe71 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 0, expected);
    }
    SUBCASE("matmul, set 5, param 1")
    {
        std::vector<uint32_t> expected = { 0xde086d85, 0x5e630878, 0x5eba5c7b };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 1, expected);
    }
}

void conv2d_test_FP32(const std::string tosaName[3],
                      const size_t tosaElements[3],
                      const std::string templateJsonCfg,
                      const std::string setStr,
                      int32_t param,
                      const std::vector<uint32_t> lastExpected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<float> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 4));
    std::vector<float> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
    check_output<float>(last_three, lastExpected);
}

TEST_CASE("positive - FP32 conv2d dot product (last 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 1, 8, 2, 4 ],
                "input_pos": 0,
                "op" : "CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 16,
                    "acc_type": "FP32",
                    "kernel": [2, 2]
                }
            },
            "weight" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "CONSTANT",
                "shape" : [ 2, 2, 2, 4 ],
                "input_pos": 1,
                "op" : "CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 16,
                    "acc_type": "FP32"
                }
            },
            "bias" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "CONSTANT",
                "shape" : [ 2 ],
                "input_pos": 2,
                "op" : "CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 16,
                    "acc_type": "FP32"
                }
            }

        }
    })";

    const std::string tosaName[3] = { "input", "weight", "bias" };
    const size_t tosaElements[3]  = { (1 * 8 * 2 * 4), (2 * 2 * 2 * 4), 2 };

    SUBCASE("conv2d, set 0, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0xbf28bfda, 0xbe99cd47 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 0, lastExpected);
    }
    SUBCASE("conv2d, set 0, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x3f648dfd, 0xbd4cb21c };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 1, lastExpected);
    }
    SUBCASE("conv2d, set 0, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 2, lastExpected);
    }
    SUBCASE("conv2d, set 1, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x5e6f0400, 0x5e2f78e5, 0x5e62318d };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 0, lastExpected);
    }
    SUBCASE("conv2d, set 1, param 1")
    {
        // NOTE: Python test script produced 0x5e6960b0 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x5e6960af, 0x5e6d0ca9, 0x5e0b8561 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 1, lastExpected);
    }
    SUBCASE("conv2d, set 1, param 2")
    {
        // NOTE: Python test script produced 0x7cf260d0, 0x7d355432 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x7cf260d1, 0x7d355431 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 2, lastExpected);
    }
    SUBCASE("conv2d, set 2, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xbc23026a, 0xbe674c38, 0x3d3d9f96 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, lastExpected);
    }
    SUBCASE("conv2d, set 2, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x3d1beb31, 0xbdc7501e, 0x3cc9f5fe };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, lastExpected);
    }
    SUBCASE("conv2d, set 2, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 2, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xbe00da48, 0x3fb5b808, 0x3e22a8bd };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 1")
    {
        // NOTE: Python test script produced 0xbfb8b240 - so off by 1
        std::vector<uint32_t> lastExpected = { 0xbd725091, 0xbfeaaf15, 0xbfb8b23f };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 2, lastExpected);
    }
    SUBCASE("conv2d, set 4, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xddcab121, 0x5d1b5485, 0xdda4bb36 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, lastExpected);
    }
    SUBCASE("conv2d, set 4, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0xdde07221, 0xde696320, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 1, lastExpected);
    }
    SUBCASE("conv2d, set 4, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 2, lastExpected);
    }
    SUBCASE("conv2d, set 5, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x5e719fb9, 0x5e6b329c, 0xdd7617d4 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 0, lastExpected);
    }
    SUBCASE("conv2d, set 5, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0xde42f57a, 0x5dd68799, 0xde2ddfcb };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 1, lastExpected);
    }
    SUBCASE("conv2d, set 5, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 2, lastExpected);
    }
}
TEST_CASE("positive - FP32 pseudo random")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 12, 3 ],
                "input_pos": 0,
                "op" : "PAD",
                "pseudo_random_info": {
                    "rng_seed": _SEED0_
                }
            },
            "input1" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 1, 3 ],
                "input_pos": 1,
                "op" : "PAD",
                "pseudo_random_info": {
                    "rng_seed": _SEED1_
                }
            }

        }
    })";

    const std::string tosaNameP0 = "input0";
    const size_t tosaElementsP0  = 12 * 3;
    const std::string tosaNameP1 = "input1";
    const size_t tosaElementsP1  = 1 * 3;

    SUBCASE("pad - same rng")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_SEED0_", "0");
        update_json_template(jsonCfg, "_SEED1_", "0");

        std::vector<float> bufferP0(tosaElementsP0);
        std::vector<float> bufferP1(tosaElementsP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaElementsP0 * 4));
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaElementsP1 * 4));
        check_output<float>(bufferP0, bufferP1);
    }

    SUBCASE("pad - different rng")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_SEED0_", "0");
        update_json_template(jsonCfg, "_SEED1_", "1000");

        std::vector<float> bufferP0(tosaElementsP0);
        std::vector<float> bufferP1(tosaElementsP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaElementsP0 * 4));
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaElementsP1 * 4));
        check_not_output<float>(bufferP0, bufferP1);
    }
}

void reduce_sum_test_FP32(const std::string tosaName,
                          const size_t tosaElements,
                          const std::string templateJsonCfg,
                          const std::string setStr,
                          const std::vector<uint32_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<float> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 4));
    // Choose different generator values to test at positions 6, 7 & 8
    std::vector<float> mid_three(buffer.begin() + 6, buffer.begin() + 9);
    check_output<float>(mid_three, expected);
}

TEST_CASE("positive - FP32 reduce_sum dot product (values 6,7 & 8)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 5, 3, 7 ],
                "input_pos": 0,
                "op" : "REDUCE_SUM",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 3,
                    "acc_type": "FP32",
                    "axis": 1
                }
            }
        }
    })";

    const std::string tosaName = "input";
    const size_t tosaElements  = 5 * 3 * 7;

    SUBCASE("reduce_sum, set 0, param 0")
    {
        std::vector<uint32_t> expected = { 0x3df2e612, 0x3f59255f, 0x0 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", expected);
    }
    SUBCASE("reduce_sum, set 1, param 0")
    {
        std::vector<uint32_t> expected = { 0x5edaa175, 0x5edb84c1, 0x5ea3c765 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("reduce_sum, set 2, param 0")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0xbe3fddf1, 0x3ef94a01 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("reduce_sum, set 3, param 0")
    {
        std::vector<uint32_t> expected = { 0x41800000, 0xc0e21e89, 0x3e77bfd7 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("reduce_sum, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0xdf0029aa, 0x3f000000, 0x3f000000 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", expected);
    }
    SUBCASE("reduce_sum, set 5, param 0")
    {
        std::vector<uint32_t> expected = { 0x5d2790c5, 0xdec3dadc, 0xdea1486e };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", expected);
    }
}

void fully_connected_test_FP32(const std::string tosaName[3],
                               const size_t tosaElements[3],
                               const std::string templateJsonCfg,
                               const std::string setStr,
                               int32_t param,
                               const std::vector<uint32_t> lastExpected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<float> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 4));
    if (param != 2)
    {
        // Get values at positions -8, -7 and -6 from the end
        std::vector<float> last_three_ish(buffer.end() - 8, buffer.end() - 5);
        check_output<float>(last_three_ish, lastExpected);
    }
    else
    {
        // Use last three as this buffer is too small
        std::vector<float> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
        check_output<float>(last_three, lastExpected);
    }
}
TEST_CASE("positive - FP32 fully_connected dot product (values -8, -7 & -6 from the end)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 6, 9 ],
                "input_pos": 0,
                "op" : "FULLY_CONNECTED",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 9,
                    "acc_type": "FP32"
                }
            },
            "weight" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "CONSTANT",
                "shape" : [ 4, 9 ],
                "input_pos": 1,
                "op" : "FULLY_CONNECTED",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 9,
                    "acc_type": "FP32"
                }
            },
            "bias" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "CONSTANT",
                "shape" : [ 4 ],
                "input_pos": 2,
                "op" : "FULLY_CONNECTED",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 9,
                    "acc_type": "FP32"
                }
            }

        }
    })";

    const std::string tosaName[3] = { "input", "weight", "bias" };
    const size_t tosaElements[3]  = { (6 * 9), (4 * 9), 4 };

    SUBCASE("fully_connected, set 0, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x3f13876f, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 0, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x3f648dfd };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 0, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 1, param 0")
    {
        // NOTE: Python test script produced 0x5e6cc8d7 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x5e531bbf, 0x5e6cc8d8, 0x5e4e2539 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 1, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x5e9870df, 0x5e9824c5, 0x5e9a898f };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 1, param 2")
    {
        // NOTE: Python test script produced 0x7dc95352 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x7d9a212a, 0x7dc95351, 0x7db7c1f2 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 2, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xba57522b, 0x3e8604b5, 0xbe861803 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 2, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x3ea90193, 0x3d4fe441, 0xbe04e014 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 2, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 0")
    {
        // NOTE: Python test script produced 0xbfe7b489 - so off by 1
        std::vector<uint32_t> lastExpected = { 0xbe947430, 0xbfe7b48a, 0xbf0a941d };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0xbe988d6d, 0xbd725091, 0xbfeaaf15 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 4, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xdce8a2e0, 0x5e327feb, 0x5ea4baf1 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 4, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0xde15a16b, 0xde9b976b };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 4, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 5, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xde8b0d70, 0xdd51465a, 0x5e57c772 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 5, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0xddde72f1, 0xde7e31ff, 0x5e0bdb32 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 5, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", 2, lastExpected);
    }
}

void avg_pool2d_test_FP32(const std::string tosaName,
                          const size_t tosaElements,
                          const std::string templateJsonCfg,
                          const std::string setStr,
                          const std::vector<uint32_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<float> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 4));
    std::vector<float> first_three(buffer.begin(), buffer.begin() + 3);
    check_output<float>(first_three, expected);
}

TEST_CASE("positive - FP32 avg_pool2d dot product (first 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 2, 6, 2, 3 ],
                "input_pos": 0,
                "op" : "AVG_POOL2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 3,
                    "acc_type": "FP32",
                    "kernel": [3, 1]
                }
            }
        }
    })";

    const std::string tosaName = "input";
    const size_t tosaElements  = 2 * 6 * 2 * 3;

    SUBCASE("avg_pool2d, set 0, param 0")
    {
        std::vector<uint32_t> expected = { 0xbf665aa4, 0xbf736bd3, 0x0 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "0", expected);
    }
    SUBCASE("avg_pool2d, set 1, param 0")
    {
        // NOTE: Python test script produced 0x5e839663,0x5e9f6894 - so off by 1
        std::vector<uint32_t> expected = { 0x5e839662, 0x5e904e86, 0x5e9f6893 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("avg_pool2d, set 2, param 0")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3ef23c13, 0x3e702703 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("avg_pool2d, set 3, param 0")
    {
        std::vector<uint32_t> expected = { 0x41800000, 0x3fee6533, 0x3f4734a9 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("avg_pool2d, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0xbf000000, 0x0 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", expected);
    }
    SUBCASE("avg_pool2d, set 5, param 0")
    {
        std::vector<uint32_t> expected = { 0x5dd5b529, 0x5e4bbbf9, 0x5eb4fe79 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "5", expected);
    }
}

TEST_CASE("positive - INT32 pseudo random")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT32",
                "input_type": "VARIABLE",
                "shape" : [ 2, 12 ],
                "input_pos": 0,
                "op" : "SCATTER",
                "pseudo_random_info": {
                    "rng_seed": 13,
                    "range": [ "-5", "5" ]
                }
            },
            "input1" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT32",
                "input_type": "VARIABLE",
                "shape" : [ 2, 10 ],
                "input_pos": 1,
                "op" : "SCATTER",
                "pseudo_random_info": {
                    "rng_seed": 14,
                    "range": [ "0", "9" ]
                }
            }

        }
    })";

    const std::string tosaNameP0 = "input0";
    const size_t tosaElementsP0  = 2 * 12;
    const std::string tosaNameP1 = "input1";
    const size_t tosaElementsP1  = 2 * 10;

    SUBCASE("scatter - int32 random")
    {
        std::string jsonCfg = templateJsonCfg;

        std::vector<int32_t> bufferP0(tosaElementsP0);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaElementsP0 * 4));
        for (auto e = bufferP0.begin(); e < bufferP0.end(); ++e)
        {
            // Check the values are within range
            bool withinRange = (*e >= -5 && *e <= 5);
            REQUIRE(withinRange);
        }
    }

    SUBCASE("scatter - int32 row shuffle")
    {
        std::string jsonCfg = templateJsonCfg;

        std::vector<int32_t> bufferP1(tosaElementsP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaElementsP1 * 4));

        std::vector<bool> set;
        for (int32_t n = 0; n < 2; ++n)
        {
            set.assign(10, false);
            for (int32_t i = 0; i < 10; ++i)
            {
                auto idx = bufferP1[i];
                // Check that the values in the buffer only occur once
                REQUIRE(!set[idx]);
                set[idx] = true;
            }
        }
    }
}
void depthwise_conv2d_test_FP16(const std::string tosaName[3],
                                const size_t tosaElements[3],
                                const std::string templateJsonCfg,
                                const std::string setStr,
                                int32_t param,
                                const std::vector<uint16_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<half_float::half> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 2));
    check_output<half_float::half>(buffer, expected);
}

TEST_CASE("positive - FP16 depthwise_conv2d dot product (first 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [1, 6, 3, 4],
                "input_pos": 0,
                "op" : "DEPTHWISE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 3,
                    "acc_type": "FP16",
                    "kernel": [1, 3]
                }
            },
            "weight" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [1, 3, 4, 2],
                "input_pos": 1,
                "op" : "DEPTHWISE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 3,
                    "acc_type": "FP16"
                }
            },
            "bias" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [ 2 ],
                "input_pos": 2,
                "op" : "DEPTHWISE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 3,
                    "acc_type": "FP16"
                }
            }

        }
    })";

    const std::string tosaName[3] = { "input", "weight", "bias" };
    const size_t tosaElements[3]  = { (1 * 6 * 3 * 4), (1 * 3 * 4 * 2), 2 };

    SUBCASE("depthwise_conv2d, set 0, param 0")
    {
        std::vector<uint16_t> expected = { 0xbb33, 0xbb9b, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 0, param 1")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x39a8 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 0, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 2, expected);
    }
    SUBCASE("depthwise_conv2d, set 1, param 0")
    {
        std::vector<uint16_t> expected = { 0x541c, 0x5482, 0x54fb };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x57ee, 0x56a2, 0x5520 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0x7005, 0x7204 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 2, expected);
    }
    SUBCASE("depthwise_conv2d, set 2, param 0")
    {
        std::vector<uint16_t> expected = { 0x3c00, 0x3c00, 0x3c00 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 2, param 1")
    {
        std::vector<uint16_t> expected = { 0x3c00, 0x3c00, 0x3c00 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 2, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 2, expected);
    }
    SUBCASE("depthwise_conv2d, set 3, param 0")
    {
        std::vector<uint16_t> expected = { 0x4c00, 0x4c00, 0x4c00 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 3, param 1")
    {
        std::vector<uint16_t> expected = { 0xcc00, 0x4c00, 0xcc00 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 3, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 2, expected);
    }
    SUBCASE("depthwise_conv2d, set 4, param 0")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 4, param 1")
    {
        std::vector<uint16_t> expected = { 0x4e14, 0xd731, 0x56e0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 4, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 2, expected);
    }
    SUBCASE("depthwise_conv2d, set 5, param 0")
    {
        std::vector<uint16_t> expected = { 0x4ead, 0x525d, 0x55a7 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 5, param 1")
    {
        std::vector<uint16_t> expected = { 0xcf61, 0x5224, 0x550b };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 5, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 2, expected);
    }
}

void transpose_conv2d_test_FP16(const std::string tosaName[3],
                                const size_t tosaElements[3],
                                const std::string templateJsonCfg,
                                const std::string setStr,
                                int32_t param,
                                const std::vector<uint16_t> lastExpected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<half_float::half> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 2));
    std::vector<half_float::half> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
    check_output<half_float::half>(last_three, lastExpected);
}

TEST_CASE("positive - FP16 transpose_conv2d dot product (last 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [1, 5, 6, 3],
                "input_pos": 0,
                "op" : "TRANSPOSE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 30,
                    "acc_type": "FP16",
                    "kernel": [5, 2]
                }
            },
            "weight" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [3, 5, 2, 3],
                "input_pos": 1,
                "op" : "TRANSPOSE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 30,
                    "acc_type": "FP16"
                }
            },
            "bias" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [3],
                "input_pos": 2,
                "op" : "TRANSPOSE_CONV2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 30,
                    "acc_type": "FP16"
                }
            }

        }
    })";

    const std::string tosaName[3] = { "input", "weight", "bias" };
    const size_t tosaElements[3]  = { (1 * 5 * 6 * 3), (3 * 5 * 2 * 3), 3 };

    SUBCASE("transpose_conv2d, set 0, param 0")
    {
        std::vector<uint16_t> expected = { 0x30e3, 0x0, 0x38e7 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 0, param 1")
    {
        std::vector<uint16_t> expected = { 0x0, 0x312a, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 0, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 2, expected);
    }
    SUBCASE("transpose_conv2d, set 1, param 0")
    {
        std::vector<uint16_t> expected = { 0x4eb3, 0x4fce, 0x4e4e };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x4e79, 0x4ed8, 0x502e };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0x6426, 0x6635, 0x680e };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 2, expected);
    }
    SUBCASE("transpose_conv2d, set 2, param 0")
    {
        std::vector<uint16_t> expected = { 0x2777, 0xafec, 0x31a5 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 2, param 1")
    {
        std::vector<uint16_t> expected = { 0x2faa, 0xac8f, 0x30b6 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 2, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 2, expected);
    }
    SUBCASE("transpose_conv2d, set 3, param 0")
    {
        std::vector<uint16_t> expected = { 0xb507, 0x4432, 0xc015 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 3, param 1")
    {
        std::vector<uint16_t> expected = { 0xb888, 0xc182, 0x26cc };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 3, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 2, expected);
    }
    SUBCASE("transpose_conv2d, set 4, param 0")
    {
        std::vector<uint16_t> expected = { 0x0, 0xcc7a, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 4, param 1")
    {
        std::vector<uint16_t> expected = { 0xcf04, 0x0, 0x5129 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 4, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 2, expected);
    }
    SUBCASE("transpose_conv2d, set 5, param 0")
    {
        std::vector<uint16_t> expected = { 0x505a, 0x4ff8, 0xd174 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 5, param 1")
    {
        std::vector<uint16_t> expected = { 0xd10e, 0xc728, 0xcd2e };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 5, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 2, expected);
    }
}

void conv3d_test_FP16(const std::string tosaName[3],
                      const size_t tosaElements[3],
                      const std::string templateJsonCfg,
                      const std::string setStr,
                      int32_t param,
                      const std::vector<uint16_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<half_float::half> buffer(tosaElements[param]);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName[param].c_str(), (void*)buffer.data(), tosaElements[param] * 2));
    check_output<half_float::half>(buffer, expected);
}

TEST_CASE("positive - FP16 conv3d dot product (first 3 values)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [1, 3, 2, 2, 3],
                "input_pos": 0,
                "op" : "CONV3D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 27,
                    "acc_type": "FP16",
                    "kernel": [3, 1, 3]
                }
            },
            "weight" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [4, 3, 1, 3, 3],
                "input_pos": 1,
                "op" : "CONV3D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 27,
                    "acc_type": "FP16"
                }
            },
            "bias" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP16",
                "input_type": "CONSTANT",
                "shape" : [ 4 ],
                "input_pos": 2,
                "op" : "CONV3D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 27,
                    "acc_type": "FP16"
                }
            }

        }
    })";

    const std::string tosaName[3] = { "input", "weight", "bias" };
    const size_t tosaElements[3]  = { (1 * 3 * 2 * 2 * 3), (4 * 3 * 1 * 3 * 3), 4 };

    SUBCASE("conv3d, set 0, param 0")
    {
        std::vector<uint16_t> expected = { 0xbb33, 0xbb9b, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 0, expected);
    }
    SUBCASE("conv3d, set 0, param 1")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x39a8 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 1, expected);
    }
    SUBCASE("conv3d, set 0, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "0", 2, expected);
    }
    SUBCASE("conv3d, set 1, param 0")
    {
        std::vector<uint16_t> expected = { 0x4e37, 0x4ed1, 0x4f87 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("conv3d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x51fe, 0x5104, 0x4fbf };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("conv3d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0x6498, 0x66e0, 0x687d };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 2, expected);
    }
    SUBCASE("conv3d, set 2, param 0")
    {
        std::vector<uint16_t> expected = { 0x3c00, 0x310c, 0x2d01 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 0, expected);
    }
    SUBCASE("conv3d, set 2, param 1")
    {
        std::vector<uint16_t> expected = { 0x3c00, 0x3099, 0x2aaf };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 1, expected);
    }
    SUBCASE("conv3d, set 2, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "2", 2, expected);
    }
    SUBCASE("conv3d, set 3, param 0")
    {
        std::vector<uint16_t> expected = { 0x4c00, 0x3f73, 0x3a3a };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 0, expected);
    }
    SUBCASE("conv3d, set 3, param 1")
    {
        std::vector<uint16_t> expected = { 0xcc00, 0xb95a, 0xb717 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 1, expected);
    }
    SUBCASE("conv3d, set 3, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "3", 2, expected);
    }
    SUBCASE("conv3d, set 4, param 0")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 0, expected);
    }
    SUBCASE("conv3d, set 4, param 1")
    {
        std::vector<uint16_t> expected = { 0x480d, 0xd0cb, 0x5095 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 1, expected);
    }
    SUBCASE("conv3d, set 4, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "4", 2, expected);
    }
    SUBCASE("conv3d, set 5, param 0")
    {
        std::vector<uint16_t> expected = { 0x490c, 0x4ccf, 0x5046 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 0, expected);
    }
    SUBCASE("conv3d, set 5, param 1")
    {
        std::vector<uint16_t> expected = { 0xc994, 0x4ca4, 0x4f9f };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 1, expected);
    }
    SUBCASE("conv3d, set 5, param 2")
    {
        std::vector<uint16_t> expected = { 0x0, 0x0, 0x0 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "5", 2, expected);
    }
}

void fft2d_test_FP32(const std::string tosaName,
                     const size_t tosaElements,
                     const std::string templateJsonCfg,
                     const std::string setStr,
                     const std::vector<uint32_t> lastExpected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_SET_", setStr);

    std::vector<float> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 4));
    // Get values at positions -8, -7 and -6 from the end
    std::vector<float> last_three_ish(buffer.end() - 8, buffer.end() - 5);

    check_output<float>(last_three_ish, lastExpected);
}

TEST_CASE("positive - FP32 fft2d dot product (values -8, -7 & -6 from the end)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "real" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 8, 2, 4 ],
                "input_pos": 0,
                "op" : "FFT2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 16,
                    "acc_type": "FP32"
                }
            },
            "imag" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 8, 2, 4 ],
                "input_pos": 1,
                "op" : "FFT2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 16,
                    "acc_type": "FP32"
                }
            }
        }
    })";

    const std::string tosaNameReal = "real";
    const std::string tosaNameImag = "imag";
    const size_t tosaElements      = 8 * 2 * 4;

    SUBCASE("fft2d, set 0, real")
    {
        std::vector<uint32_t> expected = { 0x0, 0x0, 0x3ee06867 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "0", expected);
    }
    SUBCASE("fft2d, set 0, imag")
    {
        std::vector<uint32_t> expected = { 0x3e6d1d36, 0x0, 0x0 };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "0", expected);
    }
    SUBCASE("fft2d, set 1, real")
    {
        // NOTE: Python test script produced 0x5e7219eb - so off by 1
        std::vector<uint32_t> expected = { 0x5e18358e, 0x5e7219ec, 0x5e2beab2 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("fft2d, set 1, imag")
    {
        std::vector<uint32_t> expected = { 0x5e71fbcc, 0x5e1bd27a, 0x5e46c84a };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("fft2d, set 2, real")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3e45cfbe, 0xbdd9e3f5 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("fft2d, set 2, imag")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0xbe40f9f7, 0x3dc63154 };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("fft2d, set 3, real")
    {
        std::vector<uint32_t> expected = { 0xc1800000, 0x3e143e55, 0xbfa541ab };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("fft2d, set 3, imag")
    {
        std::vector<uint32_t> expected = { 0xc1800000, 0x3edd3d0d, 0x3f204cb9 };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("fft2d, set 4, real")
    {
        std::vector<uint32_t> expected = { 0x0, 0x5d42dcdd, 0x0 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "4", expected);
    }
    SUBCASE("fft2d, set 4, imag")
    {
        std::vector<uint32_t> expected = { 0x0, 0xde0ac4fe, 0x0 };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "4", expected);
    }
    SUBCASE("fft2d, set 5, real")
    {
        std::vector<uint32_t> expected = { 0x5cb9bbd4, 0x5d8c0c21, 0x5daa1928 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "5", expected);
    }
    SUBCASE("fft2d, set 5, imag")
    {
        std::vector<uint32_t> expected = { 0x5e708eb3, 0x5e2c1a78, 0x5ddbbc3f };
        fft2d_test_FP32(tosaNameImag, tosaElements, templateJsonCfg, "5", expected);
    }
}

TEST_CASE("positive - FP32 rfft2d dot product (values -8, -7 & -6 from the end)")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "real" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 4, 2, 4 ],
                "input_pos": 0,
                "op" : "FFT2D",
                "dot_product_info": {
                    "s": _SET_,
                    "ks": 8,
                    "acc_type": "FP32"
                }
            }
        }
    })";

    const std::string tosaNameReal = "real";
    const size_t tosaElements      = 4 * 2 * 4;

    SUBCASE("rfft2d, set 0, real")
    {
        std::vector<uint32_t> expected = { 0xbe14f2f5, 0xbdb6fe4d, 0x3f30b473 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "0", expected);
    }
    SUBCASE("rfft2d, set 1, real")
    {
        // NOTE: Python test script produced 0x5e7219eb - so off by 1
        std::vector<uint32_t> expected = { 0x5e490017, 0x5e57dd30, 0x5e992496 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("rfft2d, set 2, real")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3e9d487a, 0xbe396b1f };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("rfft2d, set 3, real")
    {
        // NOTE: Python test script produced 0xbf5a96b5 - so off by 1
        std::vector<uint32_t> expected = { 0x41800000, 0xc06617b5, 0xbf5a96b6 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("rfft2d, set 4, real")
    {
        std::vector<uint32_t> expected = { 0x0, 0x0, 0x0 };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "4", expected);
    }
    SUBCASE("rfft2d, set 5, real")
    {
        std::vector<uint32_t> expected = { 0xdd3f6b86, 0xde49ecfd, 0x5e0be03d };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "5", expected);
    }
}

TEST_CASE("positive - FP16 full range")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "FULL_RANGE",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [ 48, 49, 47 ],
                "input_pos": 0,
                "op" : "CEIL",
                "full_range_info": {
                    "start_val": _START_
                }
            }
        }
    })";

    const std::string tosaName = "input0";
    const size_t tosaElements  = 48 * 49 * 47;

    SUBCASE("ceil - startVal 0")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_START_", "0");

        std::vector<half_float::half> buffer(tosaElements);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 2));
        std::vector<uint16_t> expected = { 0, 1, 2 };
        check_output<half_float::half>(buffer, expected);

        std::vector<half_float::half> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
        // To calculate last_expected: last value = tosaElements % 65535 - 1 + startVal
        std::vector<uint16_t> last_expected = { 45005, 45006, 45007 };
        check_output<half_float::half>(last_three, last_expected);
    }
    SUBCASE("ceil - startVal 100")
    {
        std::string jsonCfg = templateJsonCfg;
        update_json_template(jsonCfg, "_START_", "100");

        std::vector<half_float::half> buffer(tosaElements);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 2));
        std::vector<uint16_t> expected = { 100, 101, 102 };
        check_output<half_float::half>(buffer, expected);

        std::vector<half_float::half> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
        // To calculate last_expected: last value = tosaElements % 65535 - 1 + startVal
        std::vector<uint16_t> last_expected = { 45105, 45106, 45107 };
        check_output<half_float::half>(last_three, last_expected);
    }
}

void fp_special_test_FP32(const std::string tosaName,
                          const size_t tosaElements,
                          const std::string templateJsonCfg,
                          const std::string opStr,
                          const std::string startIndexStr,
                          const std::vector<uint32_t> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_OP_", opStr);
    update_json_template(jsonCfg, "_START_", startIndexStr);

    std::vector<float> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 4));
    check_output<float>(buffer, expected);
}

TEST_CASE("positive - FP32 FP Special")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "FP_SPECIAL",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 5, 6, 7 ],
                "input_pos": 0,
                "op" : "_OP_",
                "fp_special_info": {
                    "start_idx": _START_
                }
            },
            "input1" : {
                "generator": "FP_SPECIAL",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 5, 6, 7 ],
                "input_pos": 1,
                "op" : "_OP_",
                "fp_special_info": {
                    "start_idx": _START_
                }
            }
        }
    })";

    const std::string tosaName0 = "input0";
    const std::string tosaName1 = "input1";
    const size_t tosaElements   = 5 * 6 * 7;

    SUBCASE("equal, input 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0x7F800000, 0x0 };
        fp_special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("equal, input 1")
    {
        std::vector<uint32_t> expected = { 0x80000000, 0xFF800000, 0x80000000 };
        fp_special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("greater, input 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0x7F800000, 0x0 };
        fp_special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "GREATER", "0", expected);
    }
    SUBCASE("greater, input 1")
    {
        std::vector<uint32_t> expected = { 0x80000000, 0xFF800000, 0x80000000 };
        fp_special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "GREATER", "0", expected);
    }
    SUBCASE("add, input 0")
    {
        std::vector<uint32_t> expected = { 0x7F7FFFFF, 0x7F800000, 0x7F7FFFFF };
        fp_special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "ADD", "0", expected);
    }
    SUBCASE("add, input 1")
    {
        std::vector<uint32_t> expected = { 0x3F800000, 0xFF800000, 0x3F800000 };
        fp_special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "ADD", "0", expected);
    }
    SUBCASE("maximum, input 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0x80000000, 0x7F800000 };
        fp_special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "MAXIMUM", "0", expected);
    }
    SUBCASE("maximum, input 1")
    {
        std::vector<uint32_t> expected = { 0x0, 0x80000000, 0x7F800000 };
        fp_special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "MAXIMUM", "0", expected);
    }
    SUBCASE("maximum, startIndex 100")
    {
        // A startIndex of 100 creates an offset in the MAXIMUM op's test data (size: 13) 100 % 13 = 9
        std::vector<uint32_t> expected = { 0x80000001, 0x3F800000, 0xBF800000 };
        fp_special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "MAXIMUM", "100", expected);
    }
}

TEST_SUITE_END();    // generate
