// Copyright (c) 2023-2025, ARM Limited.
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
#include "test_utils.h"

#include <doctest.h>

#include <array>
#include <numeric>
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
std::string numbers_to_string(const std::vector<T> numbers)
{
    std::stringstream numstr;
    for (size_t idx = 0; idx < numbers.size(); ++idx)
    {
        if constexpr (std::is_same_v<T, int8_t>)
        {
            numstr << static_cast<int32_t>(numbers[idx]);
        }
        else
        {
            numstr << numbers[idx];
        }
        if (idx < numbers.size() - 1)
        {
            numstr << ", ";
        }
    }
    return numstr.str();
}

template <typename T>
void check_value(bool match, T result, T expected, uint32_t idx)
{
    std::stringstream msg;
    msg << "index: " << idx << " expected: 0x" << std::hex << uint32_t(expected) << " got: 0x" << uint32_t(result);
    if (match)
    {
        REQUIRE_MESSAGE(expected == result, msg.str());
    }
    else
    {
        REQUIRE_MESSAGE(expected != result, msg.str());
    }
}

// Bit exact checker function - binary uint32 "expected" vector
template <typename T>
void check_output(const std::vector<T>& results, const std::vector<uint32_t>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        if constexpr (sizeof(T) == 4)
        {
            uint32_t r;
            std::memcpy(&r, &results[idx], sizeof(T));
            check_value(true, *(uint32_t*)&results[idx], expected[idx], idx);
        }
        else
        {
            REQUIRE_MESSAGE(false, "INTERNAL ERROR: Types not supported by check_output()");
        }
    }
}

// Bit exact checker function - binary uint16 "expected" vector
template <typename T>
void check_output(const std::vector<T>& results, const std::vector<uint16_t>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        if constexpr (sizeof(T) == 2)
        {
            uint16_t r;
            std::memcpy(&r, &results[idx], sizeof(T));
            check_value(true, r, expected[idx], idx);
        }
        else
        {
            REQUIRE_MESSAGE(false, "INTERNAL ERROR: Types not supported by check_output()");
        }
    }
}

// Bit exact checker function
template <typename T>
void check_output(const std::vector<T>& results, const std::vector<T>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        if constexpr (sizeof(T) == 4)
        {
            uint32_t r, e;
            std::memcpy(&r, &results[idx], sizeof(T));
            std::memcpy(&e, &expected[idx], sizeof(T));
            check_value(true, r, e, idx);
        }
        else if constexpr (sizeof(T) == 2)
        {
            uint16_t r, e;
            std::memcpy(&r, &results[idx], sizeof(T));
            std::memcpy(&e, &expected[idx], sizeof(T));
            check_value(true, r, e, idx);
        }
        else if constexpr (sizeof(T) == 1)
        {
            // Don't need to perform memcpy to avoid undefined behaviour as char type
            check_value(true, *(uint8_t*)&results[idx], *(uint8_t*)&expected[idx], idx);
        }
        else
        {
            REQUIRE_MESSAGE(false, "INTERNAL ERROR: Types not supported by check_output()");
        }
    }
}

// Difference (bit exact) checker function
template <typename T>
void check_not_output(const std::vector<T>& results, const std::vector<T>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        if constexpr (sizeof(T) == 4)
        {
            uint32_t r, e;
            std::memcpy(&r, &results[idx], sizeof(T));
            std::memcpy(&e, &expected[idx], sizeof(T));
            check_value(false, r, e, idx);
        }
        else
        {
            REQUIRE_MESSAGE(false, "INTERNAL ERROR: Types not supported by check_not_output()");
        }
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
        std::vector<uint32_t> expected = { 0xdf0a6310, 0xdf114c5a, 0x5f07324b };
        matmul_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("matmul, set 1, param 1")
    {
        std::vector<uint32_t> expected = { 0x5ef54579, 0xdec1c31e, 0xdf06758a };
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
        // NOTE: Python test script produced 0x5e344528 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x5e4abec5, 0x5e344527, 0x5e684251 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 0, lastExpected);
    }
    SUBCASE("conv2d, set 1, param 1")
    {
        // NOTE: Python test script produced 0x5e1b7bf8, 0x5e0ebaf1 - so off by 1
        std::vector<uint32_t> lastExpected = { 0x5e1b7bf7, 0x5e77c9f4, 0x5e0ebaf2 };
        conv2d_test_FP32(tosaName, tosaElements, templateJsonCfg, "1", 1, lastExpected);
    }
    SUBCASE("conv2d, set 1, param 2")
    {
        std::vector<uint32_t> lastExpected = { 0xfd341535, 0x7d582f77 };
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
        std::vector<uint32_t> expected = { 0xdedb5737, 0xdea85629, 0xded388af };
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
        std::vector<uint32_t> expected = { 0xdeefb178, 0xdefba9f8, 0x5eea2ac9 };
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

void test_int4_check(std::vector<int8_t> buffer, size_t elements, int8_t min, int8_t max)
{
    size_t index = 0;
    for (auto e = buffer.begin(); e < buffer.end(); ++e)
    {
        // Check the first value is within range
        int8_t v0        = (int8_t)(*e << 4) >> 4;
        bool withinRange = (v0 >= min && v0 <= max);
        std::stringstream msg;
        msg << "Index " << index << " (low half)"
            << ": " << int32_t(v0) << " not in range (" << int32_t(min) << " -> " << int32_t(max) << ")";
        REQUIRE_MESSAGE(withinRange, msg.str());

        // Check the second value is within range
        int8_t v1 = *e >> 4;
        if (index + 1 < elements)
        {
            bool withinRange = (v1 >= min && v1 <= max);
            std::stringstream msg;
            msg << "Index " << index << " (high half)"
                << ": " << int32_t(v1) << " not in range (" << int32_t(min) << " -> " << int32_t(max) << ")";
            REQUIRE_MESSAGE(withinRange, msg.str());
        }
        index += 2;
    }
}

TEST_CASE("positive - INT4")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "const0": {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT4",
                "input_type": "CONSTANT",
                "shape" : [ 3, 10 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 13,
                    "range": [ "-6", "6" ]
                }
            },
            "const1": {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT4",
                "input_type": "CONSTANT",
                "shape" : [ ],
                "input_pos": 1,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 14,
                    "range": [ "1", "7" ]
                }
            },
            "const2": {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT4",
                "input_type": "CONSTANT",
                "shape" : [ 3, 3, 3 ],
                "input_pos": 1,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 14,
                    "range": [ "-7", "-1" ]
                }
            },
            "const3": {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT4",
                "input_type": "CONSTANT",
                "shape" : [ 2, 4, 4 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 31
                }
            },
            "input4": {
                "generator": "SPECIAL",
                "data_type": "INT4",
                "input_type": "VARIABLE",
                "shape": [ 10, 2, 3 ],
                "input_pos": 1,
                "op": "CONV2D",
                "special_info": {
                    "start_idx": 1,
                    "special_test_set": "ALL_MAX_VALUES"
                }
            }
        }
    })";

    const std::string tosaNameP0  = "const0";
    const size_t tosaElementsP0   = 3 * 10;
    const size_t tosaPackedSizeP0 = (tosaElementsP0 + 1) / 2;
    const std::string tosaNameP1  = "const1";
    const size_t tosaElementsP1   = 1;
    const size_t tosaPackedSizeP1 = (tosaElementsP1 + 1) / 2;
    const std::string tosaNameP2  = "const2";
    const size_t tosaElementsP2   = 3 * 3 * 3;
    const size_t tosaPackedSizeP2 = (tosaElementsP2 + 1) / 2;
    const std::string tosaNameP3  = "const3";
    const size_t tosaElementsP3   = 2 * 4 * 4;
    const size_t tosaPackedSizeP3 = (tosaElementsP3 + 1) / 2;
    const std::string tosaNameP4  = "input4";
    const size_t tosaElementsP4   = 10 * 2 * 3;
    const size_t tosaPackedSizeP4 = (tosaElementsP4 + 1) / 2;

    SUBCASE("int4 random - rank 2 - even elements")
    {
        std::vector<int8_t> bufferP0(tosaPackedSizeP0);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaPackedSizeP0));
        test_int4_check(bufferP0, tosaElementsP0, -6, 6);
    }

    SUBCASE("int4 random - rank 0 - odd elements")
    {
        std::vector<int8_t> bufferP1(tosaPackedSizeP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaPackedSizeP1));
        test_int4_check(bufferP1, tosaElementsP1, 1, 7);
    }

    SUBCASE("int4 random - rank 3 - odd elements")
    {
        std::vector<int8_t> bufferP2(tosaPackedSizeP2);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP2.c_str(), (void*)bufferP2.data(), tosaPackedSizeP2));
        test_int4_check(bufferP2, tosaElementsP2, -7, -1);
    }

    SUBCASE("int4 random - no range set")
    {
        std::vector<int8_t> bufferP3(tosaPackedSizeP3);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP3.c_str(), (void*)bufferP3.data(), tosaPackedSizeP3));
        test_int4_check(bufferP3, tosaElementsP3, -7, 7);
    }

    SUBCASE("int4 special - all max values")
    {
        std::vector<int8_t> bufferP4(tosaPackedSizeP4);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP4.c_str(), (void*)bufferP4.data(), tosaPackedSizeP4));
        test_int4_check(bufferP4, tosaElementsP4, 7, 7);
    }
}

TEST_CASE("positive - BOOL pseudo random")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "const0" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "BOOL",
                "input_type": "CONST",
                "shape" : [ 4, 3 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 10,
                    "range": [ "0", "1" ]
                }
            },
            "const1" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "BOOL",
                "input_type": "CONST",
                "shape" : [ 12, 3 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 30
                }
            }
        }
    })";

    const std::string tosaNameP0 = "const0";
    const size_t tosaElementsP0  = 4 * 3;
    const std::string tosaNameP1 = "const1";
    const size_t tosaElementsP1  = 12 * 3;

    SUBCASE("BOOL random - range set")
    {
        std::vector<int8_t> bufferP0(tosaElementsP0);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaElementsP0));
        for (auto e = bufferP0.begin(); e < bufferP0.end(); ++e)
        {
            // Check the values are within range
            bool withinRange = (*e >= 0 && *e <= 1);
            REQUIRE(withinRange);
        }
    }
    SUBCASE("BOOL random - no range set")
    {
        std::vector<int8_t> bufferP1(tosaElementsP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaElementsP1));
        for (auto e = bufferP1.begin(); e < bufferP1.end(); ++e)
        {
            // Check the values are within range
            bool withinRange = (*e >= 0 && *e <= 1);
            REQUIRE(withinRange);
        }
    }
}

void test_int48_check(std::vector<int8_t> buffer, size_t elements, int64_t min, int64_t max)
{
    int32_t byte_pos  = 0;
    int64_t value     = 0;
    uint64_t* val_u64 = reinterpret_cast<uint64_t*>(&value);
    for (size_t idx = 0; idx < buffer.size(); ++idx)
    {
        uint8_t byte_val = static_cast<uint8_t>(buffer[idx]);
        auto shift       = byte_pos * 8;
        *val_u64 += (static_cast<uint64_t>(byte_val) << shift);
        byte_pos++;
        // printf("byte %d: %d / v: %lu -> %ld\n", byte_pos, byte_val, *val_u64, value);

        if (byte_pos == 6)
        {
            // Sign extend by shifting up to the top and then shifting back
            *val_u64 <<= 16;
            value >>= 16;
            // Check the values are within range
            // printf("Final value: %lu -> %ld\n",  *val_u64, value);
            bool withinRange = (value >= min && value <= max);
            std::stringstream msg;
            msg << "Index " << idx / 6 << " (byte " << idx << ")"
                << ": " << value << " not in range (" << min << " -> " << max << ")";
            REQUIRE_MESSAGE(withinRange, msg.str());

            byte_pos = 0;
            value    = 0;
        }
    }
    // Fail if we end up with unused bytes
    REQUIRE(byte_pos == 0);
}

TEST_CASE("positive - INT48")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "const0" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT48",
                "input_type": "CONST",
                "shape" : [ 10 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 17
                }
            },
            "const1" : {
                "generator": "PSEUDO_RANDOM",
                "data_type": "INT48",
                "input_type": "CONST",
                "shape" : [ 5, 8, 3 ],
                "input_pos": 0,
                "op" : "CONST",
                "pseudo_random_info": {
                    "rng_seed": 34,
                    "range": [ "-1000000", "-5000" ]
                }
            },
            "input2": {
                "generator": "SPECIAL",
                "data_type": "INT48",
                "input_type": "VARIABLE",
                "shape": [ 10, 2, 3 ],
                "input_pos": 1,
                "op": "CONV2D",
                "special_info": {
                    "start_idx": 1,
                    "special_test_set": "ALL_MAX_VALUES"
                }
            }        }
    })";

    const std::string tosaNameP0 = "const0";
    const size_t tosaElementsP0  = 10;
    const size_t tosaSizeP0      = tosaElementsP0 * 6;
    const std::string tosaNameP1 = "const1";
    const size_t tosaElementsP1  = 5 * 8 * 3;
    const size_t tosaSizeP1      = tosaElementsP1 * 6;
    const std::string tosaNameP2 = "input2";
    const size_t tosaElementsP2  = 10 * 2 * 3;
    const size_t tosaSizeP2      = tosaElementsP2 * 6;

    const int64_t max    = +(static_cast<int64_t>(1) << 47) - 1;
    const int64_t lowest = -(static_cast<int64_t>(1) << 47);

    SUBCASE("INT48 random - no range set")
    {
        std::vector<int8_t> bufferP0(tosaSizeP0);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP0.c_str(), (void*)bufferP0.data(), tosaSizeP0));
        test_int48_check(bufferP0, tosaSizeP0, lowest, max);
    }
    SUBCASE("INT48 random - negative range set")
    {
        std::vector<int8_t> bufferP1(tosaSizeP1);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP1.c_str(), (void*)bufferP1.data(), tosaSizeP1));
        test_int48_check(bufferP1, tosaSizeP1, -1000000L, -5000L);
    }
    SUBCASE("INT48 special - all max values")
    {
        std::vector<int8_t> bufferP2(tosaSizeP2);
        REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaNameP2.c_str(), (void*)bufferP2.data(), tosaSizeP2));
        test_int48_check(bufferP2, tosaSizeP2, max, max);
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
        std::vector<uint16_t> expected = { 0xd77d, 0xd7dc, 0x5750 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("depthwise_conv2d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x56a2, 0xd53e, 0xd746 };
        depthwise_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("depthwise_conv2d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0xf1f9, 0x732c };
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
        std::vector<uint16_t> expected = { 0xcdd5, 0xcdeb, 0xd0c0 };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("transpose_conv2d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x5026, 0x4f9f, 0x506b };
        transpose_conv2d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("transpose_conv2d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0xe62b, 0x6767, 0xe449 };
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
        std::vector<uint16_t> expected = { 0xd1a9, 0xd1f1, 0x5187 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 0, expected);
    }
    SUBCASE("conv3d, set 1, param 1")
    {
        std::vector<uint16_t> expected = { 0x5104, 0xcfed, 0xd180 };
        conv3d_test_FP16(tosaName, tosaElements, templateJsonCfg, "1", 1, expected);
    }
    SUBCASE("conv3d, set 1, param 2")
    {
        std::vector<uint16_t> expected = { 0xe6d4, 0x6819, 0xe4be };
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
        std::vector<uint32_t> expected = { 0xde212c17, 0x5e721acc, 0x5e148e4d };
        fft2d_test_FP32(tosaNameReal, tosaElements, templateJsonCfg, "1", expected);
    }
    SUBCASE("fft2d, set 1, imag")
    {
        // NOTE: Python test script produced 0x5e3dbd3a, 0xde55aa26 - so off by 1
        std::vector<uint32_t> expected = { 0x5e3dbd39, 0xde55aa27, 0x5e6966c6 };
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
        // NOTE: Python test script produced 0xde4d5989 - so off by 1
        std::vector<uint32_t> expected = { 0xde4d598a, 0xde881df8, 0xdea7b1e7 };
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
        // TODO: Re-enable subnorm testing - (0, 1, 2)
        std::vector<uint16_t> expected = { 0, 0, 0 };
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
        // TODO: Re-enable subnorm testing - (100, 101, 102)
        std::vector<uint16_t> expected = { 0, 0, 0 };
        check_output<half_float::half>(buffer, expected);

        std::vector<half_float::half> last_three(buffer.end() - std::min<int>(3, buffer.size()), buffer.end());
        // To calculate last_expected: last value = tosaElements % 65535 - 1 + startVal
        std::vector<uint16_t> last_expected = { 45105, 45106, 45107 };
        check_output<half_float::half>(last_three, last_expected);
    }
}

void special_test_FP32(const std::string tosaName,
                       const size_t tosaElements,
                       const std::string templateJsonCfg,
                       const std::string opStr,
                       const std::string startIndexStr,
                       const std::vector<std::pair<float, float>> expected)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_OP_", opStr);
    update_json_template(jsonCfg, "_START_", startIndexStr);

    std::vector<float> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 4));
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        std::stringstream msg;
        msg << "index: " << idx << " expected between: " << expected[idx].first << " and: " << expected[idx].second
            << ", but got: " << buffer[idx];
        if (std::isnan(expected[idx].first) || std::isnan(expected[idx].second))
        {
            REQUIRE_MESSAGE((std::isnan(expected[idx].first) && std::isnan(expected[idx].second)),
                            "Incorrect test - cannot have range that includes NaN, both values must be NaN");
            REQUIRE_MESSAGE(std::isnan(buffer[idx]), msg.str());
        }
        else
        {
            // We check for sign to properly cover the sign of zero in floating point types.
            const bool signMatches = std::signbit(buffer[idx]) == std::signbit(expected[idx].first) ||
                                     std::signbit(buffer[idx]) == std::signbit(expected[idx].second);
            const bool valueInRange  = (buffer[idx] >= expected[idx].first && buffer[idx] <= expected[idx].second);
            const bool correctOutput = signMatches && valueInRange;
            REQUIRE_MESSAGE(correctOutput, msg.str());
        }
    }
}

TEST_CASE("positive - FP32 FP Special")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "SPECIAL",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 5, 6, 7 ],
                "input_pos": 0,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_
                }
            },
            "input1" : {
                "generator": "SPECIAL",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 5, 6, 7 ],
                "input_pos": 1,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_
                }
            }
        }
    })";

    const std::string tosaName0 = "input0";
    const std::string tosaName1 = "input1";
    const size_t tosaElements   = 5 * 6 * 7;
    const float inf             = std::numeric_limits<float>::infinity();
    const float min             = std::numeric_limits<float>::min();
    const float max             = std::numeric_limits<float>::max();
    const float ulpmax          = 3.777893186295716e+22;    // max - nextafter(max, 0.0)
    const float mindenorm       = std::numeric_limits<float>::denorm_min();
    const float nanFloat        = std::nanf("1");

    SUBCASE("equal, input 0")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },           { -inf, -inf }, { inf, inf },           { -inf, -inf },
            { -0.0, -0.0 },         { 0.0, 0.0 },   { nanFloat, nanFloat }, { -mindenorm, max },
            { nanFloat, nanFloat }, { inf, inf },   { nanFloat, nanFloat }, { -inf, -inf },
            { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("equal, input 1")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },   { -inf, -inf },         { -inf, -inf },         { inf, inf }, { 0.0, 0.0 },
            { -0.0, -0.0 }, { -mindenorm, max },    { nanFloat, nanFloat }, { inf, inf }, { nanFloat, nanFloat },
            { -inf, -inf }, { nanFloat, nanFloat }, { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("greater, input 0")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },           { -inf, -inf }, { inf, inf },           { -inf, -inf },
            { -0.0, -0.0 },         { 0.0, 0.0 },   { nanFloat, nanFloat }, { -mindenorm, max },
            { nanFloat, nanFloat }, { inf, inf },   { nanFloat, nanFloat }, { -inf, -inf },
            { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("greater, input 1")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },   { -inf, -inf },         { -inf, -inf },         { inf, inf }, { 0.0, 0.0 },
            { -0.0, -0.0 }, { -mindenorm, max },    { nanFloat, nanFloat }, { inf, inf }, { nanFloat, nanFloat },
            { -inf, -inf }, { nanFloat, nanFloat }, { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("greater_equal, input 0")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },           { -inf, -inf }, { inf, inf },           { -inf, -inf },
            { -0.0, -0.0 },         { 0.0, 0.0 },   { nanFloat, nanFloat }, { -mindenorm, max },
            { nanFloat, nanFloat }, { inf, inf },   { nanFloat, nanFloat }, { -inf, -inf },
            { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("greater_equal, input 1")
    {
        std::vector<std::pair<float, float>> expected = {
            { inf, inf },   { -inf, -inf },         { -inf, -inf },         { inf, inf }, { 0.0, 0.0 },
            { -0.0, -0.0 }, { -mindenorm, max },    { nanFloat, nanFloat }, { inf, inf }, { nanFloat, nanFloat },
            { -inf, -inf }, { nanFloat, nanFloat }, { nanFloat, nanFloat }
        };
        special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "EQUAL", "0", expected);
    }
    SUBCASE("add, input 0")
    {
        std::vector<std::pair<float, float>> expected = { { ulpmax, max }, { -max, -max }, { max, max } };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "ADD", "0", expected);
    }
    SUBCASE("add, input 1")
    {
        std::vector<std::pair<float, float>> expected = { { max, max }, { -max, -ulpmax }, { ulpmax, ulpmax } };
        special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "ADD", "0", expected);
    }
    SUBCASE("maximum, input 0")
    {
        std::vector<std::pair<float, float>> expected = { { 0.0, 0.0 }, { inf, inf }, { min, min } };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "MAXIMUM", "0", expected);
    }
    SUBCASE("maximum, input 1")
    {
        std::vector<std::pair<float, float>> expected = { { -0.0, -0.0 }, { -inf, -inf }, { -min, -min } };
        special_test_FP32(tosaName1, tosaElements, templateJsonCfg, "MAXIMUM", "0", expected);
    }
    SUBCASE("maximum, startIndex 100")
    {
        // A startIndex of 100 creates an offset in the MAXIMUM op's test data (size: 6) 98 % 6 = 2
        std::vector<std::pair<float, float>> expected = { { min, min }, { max, max }, { 1.0, max } };
        special_test_FP32(tosaName0, tosaElements, templateJsonCfg, "MAXIMUM", "98", expected);
    }
}

enum valueType
{
    Float,
    Integer,
    OddInteger,
    EvenInteger
};

void special_test_FP16(const std::string tosaName,
                       const size_t tosaElements,
                       const std::string templateJsonCfg,
                       const std::string opStr,
                       const std::string startIndexStr,
                       const std::vector<std::pair<half_float::half, half_float::half>> expected,
                       const std::vector<valueType> expectedValueType)
{
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_OP_", opStr);
    update_json_template(jsonCfg, "_START_", startIndexStr);

    std::vector<half_float::half> buffer(tosaElements);
    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaElements * 2));
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        std::stringstream msg;
        msg << "index: " << idx << " expected between: " << expected[idx].first << " and: " << expected[idx].second
            << ", but got: " << buffer[idx];
        if (std::isnan(expected[idx].first) || std::isnan(expected[idx].second))
        {
            REQUIRE_MESSAGE((std::isnan(expected[idx].first) && std::isnan(expected[idx].second)),
                            "Incorrect test - cannot have range that includes NaN, both values must be NaN");
            REQUIRE_MESSAGE(std::isnan(buffer[idx]), msg.str());
        }
        else
        {
            const bool signMatches = std::signbit(double(buffer[idx])) == std::signbit(double(expected[idx].first)) ||
                                     std::signbit(double(buffer[idx])) == std::signbit(double(expected[idx].second));
            const bool valueInRange  = (buffer[idx] >= expected[idx].first && buffer[idx] <= expected[idx].second);
            const bool correctOutput = signMatches && valueInRange;
            REQUIRE_MESSAGE(correctOutput, msg.str());

            if (expectedValueType[idx] != Float)
            {
                std::stringstream imsg;
                imsg << "index: " << idx << " got: " << buffer[idx] << " but expected an integer";
                bool isInteger = buffer[idx] == round(buffer[idx]);
                REQUIRE_MESSAGE(isInteger, imsg.str());

                if (expectedValueType[idx] == OddInteger)
                {
                    half_float::half halfValue = buffer[idx] / half_float::half(2.0);
                    bool isOdd                 = halfValue != round(halfValue);
                    imsg << " that is odd";
                    REQUIRE_MESSAGE(isOdd, imsg.str());
                }
                if (expectedValueType[idx] == EvenInteger)
                {
                    half_float::half halfValue = buffer[idx] / half_float::half(2.0);
                    bool isEven                = halfValue == round(halfValue);
                    imsg << " that is even";
                    REQUIRE_MESSAGE(isEven, imsg.str());
                }
            }
        }
    }
}

TEST_CASE("positive - FP16 FP Special")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "SPECIAL",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [ 3, 6, 4 ],
                "input_pos": 0,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_
                }
            },
            "input1" : {
                "generator": "SPECIAL",
                "data_type": "FP16",
                "input_type": "VARIABLE",
                "shape" : [ 3, 6, 4 ],
                "input_pos": 1,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_
                }
            }
        }
    })";

    const std::string tosaName0       = "input0";
    const std::string tosaName1       = "input1";
    const size_t tosaElements         = 3 * 6 * 4;
    const half_float::half max        = std::numeric_limits<half_float::half>::max();
    const half_float::half min        = std::numeric_limits<half_float::half>::min();
    const half_float::half pythagoras = half_float::half(1.41421);
    const half_float::half two        = half_float::half(2.0);
    const half_float::half ulpmax     = half_float::half(32.0);    // max - nextafter(max, 0.0)
    const half_float::half inf        = std::numeric_limits<half_float::half>::infinity();

    SUBCASE("pow, input 0")
    {
        std::vector<std::pair<half_float::half, half_float::half>> expected = { { min, max },
                                                                                { min, max },
                                                                                { max, max } };
        std::vector<valueType> expectedValueType                            = { Float, Float, Float };
        special_test_FP16(tosaName0, tosaElements, templateJsonCfg, "POW", "2", expected, expectedValueType);
    }
    SUBCASE("pow, input 1")
    {
        std::vector<std::pair<half_float::half, half_float::half>> expected = { { pythagoras, pythagoras },
                                                                                { -pythagoras, -pythagoras },
                                                                                { two, max } };
        std::vector<valueType> expectedValueType                            = { Float, Float, Float };
        special_test_FP16(tosaName1, tosaElements, templateJsonCfg, "POW", "2", expected, expectedValueType);
    }
    SUBCASE("sub, input 0")
    {
        std::vector<std::pair<half_float::half, half_float::half>> expected = { { max, max },
                                                                                { -ulpmax, -ulpmax },
                                                                                { inf, inf } };
        std::vector<valueType> expectedValueType                            = { Float, Float, Float };
        special_test_FP16(tosaName0, tosaElements, templateJsonCfg, "SUB", "2", expected, expectedValueType);
    }
    SUBCASE("sub, input 1")
    {
        std::vector<std::pair<half_float::half, half_float::half>> expected = { { -ulpmax, -ulpmax },
                                                                                { max, max },
                                                                                { inf, inf } };
        std::vector<valueType> expectedValueType                            = { Float, Float, Float };
        special_test_FP16(tosaName1, tosaElements, templateJsonCfg, "SUB", "2", expected, expectedValueType);
    }
}

template <typename INT_TYPE, typename StorageType>
void special_generate_INT(const std::string tosaName,
                          const size_t tosaElements,
                          const std::string opStr,
                          const std::string startIndexStr,
                          const std::string testSetStr,
                          std::vector<StorageType>& buffer)
{
    std::string jsonCfg = R"({
        "tensors" : {
            "input0" : {
                "generator": "SPECIAL",
                "data_type": "_INT_TYPE_",
                "unsigned_data": _UNSIGNED_,
                "input_type": "VARIABLE",
                "shape" : [ 3, 6, 4 ],
                "input_pos": 0,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_,
                    "special_test_set": "_TEST_SET_"
                }
            },
            "input1" : {
                "generator": "SPECIAL",
                "data_type": "_INT_TYPE_",
                "unsigned_data": _UNSIGNED_,
                "input_type": "VARIABLE",
                "shape" : [ 3, 6, 4 ],
                "input_pos": 1,
                "op" : "_OP_",
                "special_info": {
                    "start_idx": _START_
                }
            },
            "const0": {
                "generator": "FULL_RANGE",
                "data_type": "_INT_TYPE_",
                "unsigned_data": _UNSIGNED_,
                "input_type": "VARIABLE",
                "shape" : [ 3, 6, 4 ],
                "input_pos": 1,
                "op" : "_OP_",
                "full_range_info": {
                    "start_val": _START_
                }
            }
        }
    })";

    ;
    DType dtype = NativeType2DType<INT_TYPE>();

    update_json_template(jsonCfg, "_OP_", opStr);
    update_json_template(jsonCfg, "_START_", startIndexStr);
    std::string dtypeStr(EnumNameDType(dtype));
    std::string unsignedStr;
    if (dtypeStr == "UINT16")
    {
        dtypeStr    = "INT16";
        unsignedStr = "true";
    }
    else if (dtypeStr == "UINT8")
    {
        dtypeStr    = "INT8";
        unsignedStr = "true";
    }
    else
    {
        unsignedStr = "false";
    }
    update_json_template(jsonCfg, "_INT_TYPE_", dtypeStr);
    update_json_template(jsonCfg, "_UNSIGNED_", unsignedStr);
    update_json_template(jsonCfg, "_TEST_SET_", testSetStr);

    size_t bytes = tosaElements * sizeof(INT_TYPE);

    REQUIRE(tgd_generate_data(jsonCfg.c_str(), tosaName.c_str(), (void*)buffer.data(), bytes));
}

template <typename INT_TYPE, typename StorageType>
void special_check_expected_INT(const std::vector<StorageType>& buffer,
                                const std::vector<std::pair<INT_TYPE, INT_TYPE>>& expected,
                                const std::string opStr,
                                const bool repeatExpected)
{
    for (size_t idx = 0; idx < buffer.size(); ++idx)
    {
        size_t exidx = idx % expected.size();
        std::stringstream msg;
        msg << opStr << " index buffer/expected: [" << idx << "] / [" << exidx
            << "] expected value between: " << int64_t(expected[exidx].first)
            << " and: " << int64_t(expected[exidx].second) << ", but got: " << int64_t(buffer[idx]);
        bool withinRange = buffer[idx] >= expected[exidx].first && buffer[idx] <= expected[exidx].second;

        REQUIRE_MESSAGE(withinRange, msg.str());

        // Have all expected values been checked if not repeated across whole buffer
        if (!repeatExpected && (exidx + 1 == expected.size()))
            break;
    }
}

template <typename INT_TYPE>
void special_test_INT(const std::string tosaName,
                      const size_t tosaElements,
                      const std::string opStr,
                      const std::string startIndexStr,
                      const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected)
{
    std::vector<INT_TYPE> buffer(tosaElements);
    special_generate_INT<INT_TYPE, INT_TYPE>(tosaName, tosaElements, opStr, startIndexStr, "DEFAULT", buffer);
    // Check a single set of expected values is present in the buffer
    special_check_expected_INT(buffer, expected, opStr, false);
}

template <typename INT_TYPE>
void full_range_test_INT(const std::string tosaName,
                         const size_t tosaElements,
                         const std::string opStr,
                         const std::string startValueStr,
                         const INT_TYPE startValue)
{
    std::vector<INT_TYPE> buffer(tosaElements);
    special_generate_INT<INT_TYPE, INT_TYPE>(tosaName, tosaElements, opStr, startValueStr, "DEFAULT", buffer);
    // Test all values in the buffer match the expected values repeated
    INT_TYPE value = startValue;
    for (size_t idx = 0; idx < buffer.size(); ++idx)
    {
        std::stringstream msg;
        msg << opStr << " index: [" << idx << "] expected: " << int64_t(value) << ", but got: " << int64_t(buffer[idx]);
        bool okay = uint64_t(buffer[idx]) == uint64_t(value);

        REQUIRE_MESSAGE(okay, msg.str());
        value++;
    }
}

template <typename INT_TYPE>
void special_test_set_INT(const std::string tosaName,
                          const size_t tosaElements,
                          const std::string opStr,
                          const std::string startIndexStr,
                          const std::string testSetStr,
                          const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected)
{
    std::vector<INT_TYPE> buffer(tosaElements);
    special_generate_INT<INT_TYPE, INT_TYPE>(tosaName, tosaElements, opStr, startIndexStr, testSetStr, buffer);
    // Test all values in the buffer match the expected values repeated
    special_check_expected_INT(buffer, expected, opStr, true);
}

// Special case for vector of bool that is packed in C, so use int8 for the buffer
template <>
void special_test_set_INT<bool>(const std::string tosaName,
                                const size_t tosaElements,
                                const std::string opStr,
                                const std::string startIndexStr,
                                const std::string testSetStr,
                                const std::vector<std::pair<bool, bool>> expected)
{
    std::vector<int8_t> buffer(tosaElements);
    special_generate_INT<bool, int8_t>(tosaName, tosaElements, opStr, startIndexStr, testSetStr, buffer);
    // Test all values in the buffer match the expected values repeated
    special_check_expected_INT(buffer, expected, opStr, true);
}

template <typename INT_TYPE>
void special_binary_coverage_INT(const size_t tosaElements, const Op op)
{
    const std::string tosaName0 = "input0";
    const std::string tosaName1 = "input1";

    // Generate values in first input
    std::vector<INT_TYPE> input0(tosaElements);
    special_generate_INT<INT_TYPE, INT_TYPE>(tosaName0, tosaElements, EnumNameOp(op), "0", "DEFAULT", input0);

    // Generate values in second input
    std::vector<INT_TYPE> input1(tosaElements);
    special_generate_INT<INT_TYPE, INT_TYPE>(tosaName1, tosaElements, EnumNameOp(op), "0", "DEFAULT", input1);

    bool hasMax    = false;
    bool hasLowest = false;
    bool hasZero   = false;

    for (size_t i = 0; i < input0.size(); i++)
    {
        // We will assume no shift in the result. Our generation algorithm doesn't know the shift.
        int64_t in0 = static_cast<int64_t>(input0[i]);
        int64_t in1 = static_cast<int64_t>(input1[i]);
        int64_t result;

        switch (op)
        {
            case Op_MUL:
                // NOTE: Ignore shift for our purposes, because we don't know the
                // shift when we generate the value.
                // TODO: update once shift starts being taken into consideration
                result = in0 * in1;
                break;
            case Op_ADD:
                result = in0 + in1;
                break;
            case Op_SUB:
                result = in0 - in1;
                break;
            case Op_INTDIV:
                result = in0 / in1;
                break;
            default:
                result = 0;
                REQUIRE_MESSAGE(false, "Error in unit test construction: unsupported op");
        }

        if (result == std::numeric_limits<INT_TYPE>::max())
            hasMax = true;
        if (result == std::numeric_limits<INT_TYPE>::lowest())
            hasLowest = true;
        if (result == 0)
            hasZero = true;

        bool withinRange =
            std::numeric_limits<INT_TYPE>::lowest() <= result && result <= std::numeric_limits<INT_TYPE>::max();
        REQUIRE(withinRange);
    }

    CHECK(hasMax);
    CHECK(hasLowest);
    CHECK(hasZero);
}

TEST_CASE_TEMPLATE("positive - INT SPECIAL", INT_TYPE, bool, int8_t, int16_t, int32_t, uint8_t, uint16_t)
{
    const std::string tosaName0 = "input0";
    const std::string tosaName1 = "input1";
    const std::string tosaName2 = "const0";
    const size_t tosaElements   = 3 * 6 * 4;

    const std::pair<INT_TYPE, INT_TYPE> zero{ 0, 0 };
    const std::pair<INT_TYPE, INT_TYPE> one{ 1, 1 };
    const std::pair<INT_TYPE, INT_TYPE> minusOne{ -1, -1 };
    const std::pair<INT_TYPE, INT_TYPE> minusTwo{ -2, -2 };
    const std::pair<INT_TYPE, INT_TYPE> max{ std::numeric_limits<INT_TYPE>::max(),
                                             std::numeric_limits<INT_TYPE>::max() };
    const std::pair<INT_TYPE, INT_TYPE> minusMax{ -std::numeric_limits<INT_TYPE>::max(),
                                                  -std::numeric_limits<INT_TYPE>::max() };
    const std::pair<INT_TYPE, INT_TYPE> lowest{ std::numeric_limits<INT_TYPE>::lowest(),
                                                std::numeric_limits<INT_TYPE>::lowest() };
    const std::pair<INT_TYPE, INT_TYPE> random{ std::numeric_limits<INT_TYPE>::lowest(),
                                                std::numeric_limits<INT_TYPE>::max() };

    // Test set
    const std::pair<INT_TYPE, INT_TYPE> smallValues{ -2, 2 };

    // Used for shift operators
    const int64_t maxShiftVal = sizeof(INT_TYPE) * 8 - 1;
    const std::pair<INT_TYPE, INT_TYPE> randShift{ 0, maxShiftVal };
    const std::pair<INT_TYPE, INT_TYPE> maxShift{ maxShiftVal, maxShiftVal };

    const std::pair<INT_TYPE, INT_TYPE> nonPositive{ std::numeric_limits<INT_TYPE>::lowest(), 0 };
    const std::pair<INT_TYPE, INT_TYPE> nonNegative{ 0, std::numeric_limits<INT_TYPE>::max() };

    // Expected default values set
    const std::vector<std::pair<INT_TYPE, INT_TYPE>> expectedDefault = { lowest,   max,    minusMax, one,
                                                                         minusOne, random, zero,     lowest };

    // Tests only available for int32
    if constexpr (std::is_same_v<INT_TYPE, int32_t>)
    {
        SUBCASE("equal,greater,greater_equal input 0")
        {
            const std::vector<std::string> operators = { "EQUAL", "GREATER", "GREATER_EQUAL" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    zero, lowest, zero, zero, random, max, random, lowest, random, zero, max,
                };
                special_test_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", expected);
            }
        }

        SUBCASE("equal,greater,greater_equal input 1")
        {
            const std::vector<std::string> operators = { "EQUAL", "GREATER", "GREATER_EQUAL" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    max, zero, lowest, random, zero, random, max, random, lowest, zero, zero, max, zero, lowest,
                };
                special_test_INT(tosaName1, tosaElements, op, "1", expected);
            }
        }

        SUBCASE("bitwise_and,or,xor input 0")
        {
            const std::vector<std::string> operators = { "BITWISE_AND", "BITWISE_OR", "BITWISE_XOR" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { max,    zero, lowest, zero,   zero,
                                                                              random, max,  random, lowest, random,
                                                                              zero,   max,  zero };
                special_test_INT(tosaName0, tosaElements, op, "0", expected);
            }
        }

        SUBCASE("bitwise_and,or,xor input 1")
        {
            const std::vector<std::string> operators = { "BITWISE_AND", "BITWISE_OR", "BITWISE_XOR" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    zero, max, zero, lowest, random, zero, random, max, random, lowest, zero, zero, max, zero, lowest,
                };
                special_test_INT(tosaName1, tosaElements, op, "0", expected);
            }
        }

        SUBCASE("maximum,minimum input 0")
        {
            const std::vector<std::string> operators = { "MAXIMUM", "MINIMUM" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    random, zero, max, zero, lowest, zero, zero, random, max, random, lowest, random, zero,
                };
                special_test_INT(tosaName0, tosaElements, op, "9", expected);
            }
        }

        SUBCASE("maximum,minimum input 1")
        {
            const std::vector<std::string> operators = { "MAXIMUM", "MINIMUM" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    lowest, zero, zero, max, zero, lowest, random, zero, random, max, random, lowest, zero, zero,
                };
                special_test_INT(tosaName1, tosaElements, op, "9", expected);
            }
        }

        SUBCASE("default input 0 int32 only")
        {
            const std::vector<std::string> operators = { "CLAMP", "BITWISE_NOT", "CLZ" };
            for (const auto& op : operators)
            {
                special_test_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", expectedDefault);
            }
        }

        SUBCASE("negate/abs input 0")
        {
            // Range to avoid overflow
            const std::vector<std::string> operators = { "ABS", "NEGATE" };
            const std::pair<INT_TYPE, INT_TYPE> randomCapped{ -std::numeric_limits<INT_TYPE>::max(),
                                                              std::numeric_limits<INT_TYPE>::max() };
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { max,          minusMax, one, minusOne,
                                                                          randomCapped, zero,     max };
            for (const auto& op : operators)
            {
                special_test_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", expected);
            }
        }
    }

    // Tests available for int32, int16 and int8
    if constexpr (std::is_same_v<INT_TYPE, int32_t> || std::is_same_v<INT_TYPE, int16_t> ||
                  std::is_same_v<INT_TYPE, int8_t>)
    {
        SUBCASE("shifts, input 0")
        {
            const std::vector<std::string> operators = { "LOGICAL_RIGHT_SHIFT", "LOGICAL_LEFT_SHIFT",
                                                         "ARITHMETIC_RIGHT_SHIFT" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { zero,   zero,   zero,   lowest, lowest,
                                                                              lowest, random, random, max,    max,
                                                                              max,    zero,   zero,   zero,   lowest };
                special_test_INT(tosaName0, tosaElements, op, "3", expected);
            }
        }

        SUBCASE("shifts, input 1")
        {
            const std::vector<std::string> operators = { "LOGICAL_RIGHT_SHIFT", "LOGICAL_LEFT_SHIFT",
                                                         "ARITHMETIC_RIGHT_SHIFT" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                    randShift, zero, maxShift, randShift, zero, maxShift, zero,      maxShift,
                    randShift, zero, maxShift, randShift, zero, maxShift, randShift,
                };
                special_test_INT(tosaName1, tosaElements, op, "3", expected);
            }
        }

        SUBCASE("mul, coverage")
        {
            special_binary_coverage_INT<INT_TYPE>(tosaElements, Op_MUL);
        }

        SUBCASE("mul, input 0")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                max,      lowest,   lowest, zero,   zero,        zero,        one, one, one, minusOne,
                minusOne, minusOne, random, random, nonNegative, nonPositive, max, max, max,
            };
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "MUL", "2", expected);
        }

        SUBCASE("mul, input 1")
        {
            // Using nonPositive for (-max, 0) because overflows will be caught by the coverage subcase
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                minusOne, zero, one,         max,         lowest, random, max,      lowest,
                random,   max,  nonNegative, nonPositive, zero,   one,    minusOne, minusOne,
            };
            special_test_INT<INT_TYPE>(tosaName1, tosaElements, "MUL", "2", expected);
        }

        SUBCASE("intdiv, coverage")
        {
            special_binary_coverage_INT<INT_TYPE>(tosaElements, Op_INTDIV);
        }

        SUBCASE("add, coverage")
        {
            special_binary_coverage_INT<INT_TYPE>(tosaElements, Op_ADD);
        }

        SUBCASE("add input 0")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                lowest, lowest, lowest,   zero, zero, nonPositive, nonNegative,
                max,    max,    minusMax, max,  max,  lowest,      lowest,
            };
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "ADD", "5", expected);
        }

        SUBCASE("add input 1")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                max, zero, nonNegative, max, lowest, max, lowest, lowest, minusMax, max, zero,
            };
            special_test_INT<INT_TYPE>(tosaName1, tosaElements, "ADD", "5", expected);
        }

        SUBCASE("sub, coverage")
        {
            special_binary_coverage_INT<INT_TYPE>(tosaElements, Op_SUB);
        }

        SUBCASE("sub input 0")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                zero, minusOne, nonNegative, nonPositive, lowest, lowest, lowest, lowest, max, max, max,
            };
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "SUB", "3", expected);
        }

        SUBCASE("sub input 1")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                max, lowest, max, lowest, lowest, zero, minusOne, nonPositive, max, zero, nonNegative, max, lowest,
            };
            special_test_INT<INT_TYPE>(tosaName1, tosaElements, "SUB", "3", expected);
        }

        SUBCASE("data_layout input 0")
        {
            const std::vector<std::string> operators = { "CONCAT", "PAD",       "RESHAPE", "REVERSE", "SLICE",
                                                         "TILE",   "TRANSPOSE", "GATHER",  "SCATTER" };
            for (const auto& op : operators)
            {
                special_test_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", expectedDefault);
            }
        }

        SUBCASE("cast input 0")
        {
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "CAST", "1", expectedDefault);
        }

        SUBCASE("rescale, input 0")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { minusTwo, minusTwo, one, one,
                                                                          lowest,   lowest,   max, max };
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "RESCALE", "0", expected);
        }

        SUBCASE("test set all zeroes")
        {
            const std::vector<std::string> operators = {
                "CONV2D", "CONV3D", "DEPTHWISE_CONV2D", "TRANSPOSE_CONV2D", "MATMUL", "REDUCE_SUM", "RESIZE"
            };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { zero };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_ZEROES", expected);
            }
        }

        SUBCASE("test set all max values")
        {
            const std::vector<std::string> operators = {
                "ARGMAX",           "CONV2D",     "CONV3D",     "DEPTHWISE_CONV2D",
                "TRANSPOSE_CONV2D", "MATMUL",     "AVG_POOL2D", "MAX_POOL2D",
                "REDUCE_MIN",       "REDUCE_MAX", "RESIZE"
            };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { max };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_MAX_VALUES", expected);
            }
        }

        SUBCASE("test set all lowest values")
        {
            const std::vector<std::string> operators = { "ARGMAX",           "CONV2D",     "CONV3D",
                                                         "DEPTHWISE_CONV2D", "AVG_POOL2D", "MAX_POOL2D",
                                                         "REDUCE_MIN",       "REDUCE_MAX", "RESIZE" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { lowest };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_LOWEST_VALUES", expected);
            }
        }

        SUBCASE("test set small values")
        {
            const std::vector<std::string> operators = { "CONV2D", "CONV3D", "DEPTHWISE_CONV2D" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { smallValues };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_SMALL_VALUES", expected);
            }
        }

        SUBCASE("test set first max then all zeroes")
        {
            const std::vector<std::string> operators = { "REDUCE_SUM" };
            for (const auto& op : operators)
            {
                const size_t startIdx = 7;
                std::vector<std::pair<INT_TYPE, INT_TYPE>> expected;
                for (size_t element = 0; element < tosaElements; ++element)
                {
                    if (element == startIdx)
                        expected.push_back(max);
                    else
                        expected.push_back(zero);
                }
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, std::to_string(startIdx),
                                               "FIRST_MAX_THEN_ZEROES", expected);
            }
        }

        SUBCASE("test set first min then all zeroes")
        {
            const std::vector<std::string> operators = { "REDUCE_SUM" };
            for (const auto& op : operators)
            {
                const size_t startIdx = 7;
                std::vector<std::pair<INT_TYPE, INT_TYPE>> expected;
                for (size_t element = 0; element < tosaElements; ++element)
                {
                    if (element == startIdx)
                        expected.push_back(lowest);
                    else
                        expected.push_back(zero);
                }
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, std::to_string(startIdx),
                                               "FIRST_LOWEST_THEN_ZEROES", expected);
            }
        }

        SUBCASE("test set first max then all minus ones")
        {
            const std::vector<std::string> operators = { "REDUCE_SUM" };
            for (const auto& op : operators)
            {
                const size_t startIdx = 7;
                std::vector<std::pair<INT_TYPE, INT_TYPE>> expected;
                for (size_t element = 0; element < tosaElements; ++element)
                {
                    if (element == startIdx)
                        expected.push_back(max);
                    else
                        expected.push_back(minusOne);
                }
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, std::to_string(startIdx),
                                               "FIRST_MAX_THEN_MINUS_ONES", expected);
            }
        }

        SUBCASE("test set first min then all plus ones")
        {
            const std::vector<std::string> operators = { "REDUCE_SUM" };
            for (const auto& op : operators)
            {
                const size_t startIdx = 7;
                std::vector<std::pair<INT_TYPE, INT_TYPE>> expected;
                for (size_t element = 0; element < tosaElements; ++element)
                {
                    if (element == startIdx)
                        expected.push_back(lowest);
                    else
                        expected.push_back(one);
                }
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, std::to_string(startIdx),
                                               "FIRST_LOWEST_THEN_PLUS_ONES", expected);
            }
        }
    }

    // Tests available for int16 and int8
    if constexpr (std::is_same_v<INT_TYPE, bool>)
    {
        SUBCASE("test set all zeroes - bool")
        {
            const std::vector<std::string> operators = { "REDUCE_ANY", "REDUCE_ALL" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { zero };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_ZEROES", expected);
            }
        }

        SUBCASE("test set all max values - bool")
        {
            const std::vector<std::string> operators = { "REDUCE_ANY", "REDUCE_ALL" };
            for (const auto& op : operators)
            {
                const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = { max };
                special_test_set_INT<INT_TYPE>(tosaName0, tosaElements, op, "1", "ALL_MAX_VALUES", expected);
            }
        }
    }

    // Tests available for int16 and int8
    if constexpr (std::is_same_v<INT_TYPE, int16_t> || std::is_same_v<INT_TYPE, int8_t>)
    {
        SUBCASE("unary & table full range const 0")
        {
            const std::vector<std::string> operators = { "ABS", "BITWISE_NOT", "NEGATE", "TABLE" };
            for (const auto& op : operators)
            {
                full_range_test_INT<INT_TYPE>(tosaName2, tosaElements, op, "5", 5);
            }
        }
    }

    // Tests available for uint16 and uint8
    if constexpr (std::is_same_v<INT_TYPE, uint16_t> || std::is_same_v<INT_TYPE, uint8_t>)
    {
        SUBCASE("rescale, input 0")
        {
            const std::vector<std::pair<INT_TYPE, INT_TYPE>> expected = {
                zero, zero, one, one, lowest, lowest, max, max
            };
            special_test_INT<INT_TYPE>(tosaName0, tosaElements, "RESCALE", "0", expected);
        }
    }
}

template <typename TYPE, typename StorageType>
void fixed_data_test(const std::vector<int8_t> values, const std::vector<int32_t> shape)
{
    std::string jsonCfg = R"({
        "tensors" : {
            "fixed0" : {
                "generator": "FIXED_DATA",
                "data_type": "_TYPE_",
                "input_type": "VARIABLE",
                "shape" : [ _SHAPE_ ],
                "input_pos": 0,
                "op" : "ADD",
                "fixed_data_info": {
                    "data": [ _DATA_ ]
                }
            }
        }
    })";

    DType dtype = NativeType2DType<TYPE>();
    update_json_template(jsonCfg, "_TYPE_", EnumNameDType(dtype));
    update_json_template(jsonCfg, "_SHAPE_", numbers_to_string(shape));
    update_json_template(jsonCfg, "_DATA_", numbers_to_string(values));

    auto elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    std::vector<StorageType> expected(elements);
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        size_t vidx   = idx % values.size();
        expected[idx] = static_cast<TYPE>(values[vidx]);
    }

    std::vector<StorageType> buffer(elements);
    size_t bytes = elements * sizeof(StorageType);

    REQUIRE(tgd_generate_data(jsonCfg.c_str(), "fixed0", (void*)buffer.data(), bytes));
    check_output<StorageType>(buffer, expected);
}

TEST_CASE_TEMPLATE("positive - Fixed Data", TYPE, bool, int8_t, int16_t, int32_t, half_float::half, float)
{
    // Testing of fixed data into high-level storage types

    if constexpr (std::is_same_v<TYPE, bool>)
    {
        // bool vectors are stored as bit masks, so special handling needed
        SUBCASE("add with all data")
        {
            const std::vector<int8_t> values = { 9, 8, 7, 6, 0, -4, -3, -2, -1 };
            const std::vector<int32_t> shape = { 3, 3 };
            fixed_data_test<TYPE, int8_t>(values, shape);
        }
        SUBCASE("add with broadcast data")
        {
            const std::vector<int8_t> values = { 7 };
            const std::vector<int32_t> shape = { 9 };
            fixed_data_test<TYPE, int8_t>(values, shape);
        }
    }
    else
    {
        SUBCASE("add with all data")
        {
            const std::vector<int8_t> values = { 9, 8, 7, 6, 0, -4, -3, -2, -1 };
            const std::vector<int32_t> shape = { 3, 3 };
            fixed_data_test<TYPE, TYPE>(values, shape);
        }
        SUBCASE("add with broadcast data")
        {
            const std::vector<int8_t> values = { 7 };
            const std::vector<int32_t> shape = { 9 };
            fixed_data_test<TYPE, TYPE>(values, shape);
        }
    }
}

TEST_SUITE_END();    // generate
