// Copyright (c) 2023, ARM Limited.
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

void check_value(bool match, uint32_t result, uint32_t expected, uint32_t idx)
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
        std::vector<uint32_t> expected = { 0x3f800000, 0x3e66ed53, 0x3f800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, expected);
    }
    SUBCASE("matmul, set 2, param 1")
    {
        std::vector<uint32_t> expected = { 0x3f800000, 0x3f800000, 0x3f800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, expected);
    }
    SUBCASE("matmul, set 3, param 0")
    {
        // NOTE: Python test script produced 0xbf256686 - so off by 1
        std::vector<uint32_t> expected = { 0x41800000, 0xbf256685, 0x41800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, expected);
    }
    SUBCASE("matmul, set 3, param 1")
    {
        std::vector<uint32_t> expected = { 0x41800000, 0x41800000, 0x41800000 };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, expected);
    }
    SUBCASE("matmul, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0xbf000000, 0x5f14e80c };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, expected);
    }
    SUBCASE("matmul, set 4, param 1")
    {
        std::vector<uint32_t> expected = { 0x5d5d0db2, 0xdf2c82a8, 0x0 };
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
        std::vector<uint32_t> lastExpected = { 0x3e7da8e9, 0x3df76a57, 0xbe338212 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, lastExpected);
    }
    SUBCASE("conv2d, set 2, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x3daabbc5, 0xbe2f8909, 0xbdb806ec };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, lastExpected);
    }
    SUBCASE("conv2d, set 2, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 2, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xbee77fe5, 0x402141c5, 0xbda1b2ed };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 1")
    {
        // NOTE: Python test script produced 0xbe9947ac - so off by 1
        std::vector<uint32_t> lastExpected = { 0x3f91e619, 0x3e9ac66b, 0xbe9947ad };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, lastExpected);
    }
    SUBCASE("conv2d, set 3, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 2, lastExpected);
    }
    SUBCASE("conv2d, set 4, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0xdd7e8575, 0x0, 0xde569ff3 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, lastExpected);
    }
    SUBCASE("conv2d, set 4, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x5e2d6921, 0x5e13a014, 0x0 };
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
        std::vector<uint32_t> expected = { 0x3f800000, 0x3e73f143, 0x3f12cef8 };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("reduce_sum, set 3, param 0")
    {
        std::vector<uint32_t> expected = { 0x41800000, 0xbe9f659e, 0xbfaca78c };
        reduce_sum_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("reduce_sum, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0x5e1e6f12, 0x3f000000, 0xbf000000 };
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
        std::vector<uint32_t> lastExpected = { 0xbcc1e987, 0xbe68efd7, 0x3db90130 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 2, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x3e069935, 0x3de3a507, 0xbe6a0c0c };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 2, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x3e57454e, 0x3b48e294, 0x3e889ece };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0xbd20e608, 0x3f91e619, 0x3e9ac66b };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 1, lastExpected);
    }
    SUBCASE("fully_connected, set 3, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x0, 0x0 };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", 2, lastExpected);
    }
    SUBCASE("fully_connected, set 4, param 0")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x5e29ad6d, 0x5e959eac };
        fully_connected_test_FP32(tosaName, tosaElements, templateJsonCfg, "4", 0, lastExpected);
    }
    SUBCASE("fully_connected, set 4, param 1")
    {
        std::vector<uint32_t> lastExpected = { 0x0, 0x5e6736d7, 0x5e44d571 };
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
        std::vector<uint32_t> expected = { 0x3f800000, 0x3e3c8d18, 0xbe813879 };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "2", expected);
    }
    SUBCASE("avg_pool2d, set 3, param 0")
    {
        // NOTE: Python test script produced 0xbf256686,0x3e1e8d3b - so off by 1
        std::vector<uint32_t> expected = { 0x41800000, 0xbf256685, 0x3e1e8d3b };
        avg_pool2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "3", expected);
    }
    SUBCASE("avg_pool2d, set 4, param 0")
    {
        std::vector<uint32_t> expected = { 0x0, 0xbf000000, 0x5ef329c7 };
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
TEST_SUITE_END();    // generate
