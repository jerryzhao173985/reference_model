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
        std::vector<uint32_t> expected = { 0x0, 0x3f000000, 0x5f14e80c };
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
TEST_SUITE_END();    // generate
