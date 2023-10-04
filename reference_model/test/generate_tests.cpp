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
#include <iostream>
#include <string>
#include <vector>

namespace
{
template <typename T>
void debug_vec_print(const std::vector<T>& vec)
{
    std::cout << "vector: ";
    for (auto v = vec.begin(); v != vec.end(); ++v)
    {
        T f = *v;
        std::cout << std::dec << f << " [" << std::hex << *(uint32_t*)&f << "] ";
    }
    std::cout << std::dec << '\n';
}

void update_json_template(std::string& str, const std::string& set)
{
    std::string find = "_SET_";
    auto pos         = str.find(find);
    while (pos != std::string::npos)
    {
        str.replace(pos, find.length(), set);
        pos = str.find(find);
    }
}

template <typename T>
void check_output(const std::vector<T>& results, const std::vector<uint32_t>& expected)
{
    for (size_t idx = 0; idx < expected.size(); ++idx)
    {
        REQUIRE_MESSAGE(expected[idx] == *(uint32_t*)&results[idx], "index: ", idx);
    }
}

}    // namespace

TEST_SUITE_BEGIN("generate");

TEST_CASE("negative - api")
{
    std::string json_cfg = R"({
        "tensors" : {
            "in1" : {
                "generator": "DOT_PRODUCT",
                "data_type": "FP32",
                "input_type": "VARIABLE",
                "shape" : [ 4, 8, 8 ],
                "input_pos": 0,
                "op" : "MATMUL",
                "dot_product_info": {
                    "s": 0,
                    "ks": 8,
                    "acc_type": "FP32"
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
        std::string invalid_json_cfg = R"({
            "tensors" : {
                "in1" : {
                    "generator": DOT_PRODUCT,
                },
            }
        })";

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(invalid_json_cfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("invalid json - mismatching name")
    {
        std::string invalidName = "notFound1";

        std::vector<float> buffer(tosaElements);
        REQUIRE_FALSE(tgd_generate_data(json_cfg.c_str(), invalidName.c_str(), (void*)buffer.data(), tosaSize));
    }
    SUBCASE("mismatching size")
    {
        size_t smallElements = 4 * 8 * 7;
        size_t smallSize     = smallElements * 4;

        std::vector<float> buffer(smallElements);
        REQUIRE_FALSE(tgd_generate_data(json_cfg.c_str(), tosaName.c_str(), (void*)buffer.data(), smallSize));
    }
}

TEST_CASE("positive - dot product")
{
    std::string template_json_cfg = R"({
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

    const std::string tosaNameP0 = "in1";
    const size_t tosaElementsP0  = 4 * 8 * 2;
    const std::string tosaNameP1 = "in2";
    const size_t tosaElementsP1  = 4 * 2 * 5;

    SUBCASE("matmul, set 0, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "0");

        std::vector<uint32_t> expected = { 0xbf665aa4, 0xbf736bd3, 0x0 };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 0, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "0");

        std::vector<uint32_t> expected = { 0x0, 0x0, 0x3f34f2dd };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 1, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "1");

        std::vector<uint32_t> expected = { 0x5e97f1b0, 0x5ea6a18e, 0x5eb811af };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 1, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "1");

        std::vector<uint32_t> expected = { 0x5f128bb1, 0x5ef54579, 0x5ebd65b8 };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 2, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "2");

        std::vector<uint32_t> expected = { 0x3f800000, 0x3e66ed53, 0x3f800000 };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 2, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "2");

        std::vector<uint32_t> expected = { 0x3f800000, 0x3f800000, 0x3f800000 };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 3, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "3");

        // NOTE: Python test script produced  0xbf256686 - so off by 1
        std::vector<uint32_t> expected = { 0x41800000, 0xbf256685, 0x41800000 };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 3, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "3");

        std::vector<uint32_t> expected = { 0x41800000, 0x41800000, 0x41800000 };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 4, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "4");

        std::vector<uint32_t> expected = { 0x0, 0x3f000000, 0x5f14e80c };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 4, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "4");

        std::vector<uint32_t> expected = { 0x5d5d0db2, 0xdf2c82a8, 0x0 };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 5, param 0")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "5");

        std::vector<uint32_t> expected = { 0x5df6c4b3, 0x5e6b4088, 0x5ed0fe71 };
        std::vector<float> buffer(tosaElementsP0);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP0.c_str(), (void*)buffer.data(), tosaElementsP0 * 4));
        check_output<float>(buffer, expected);
    }
    SUBCASE("matmul, set 5, param 1")
    {
        std::string json_cfg = template_json_cfg;
        update_json_template(json_cfg, "5");

        std::vector<uint32_t> expected = { 0xde086d85, 0x5e630878, 0x5eba5c7b };
        std::vector<float> buffer(tosaElementsP1);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaNameP1.c_str(), (void*)buffer.data(), tosaElementsP1 * 4));
        check_output<float>(buffer, expected);
    }
}
TEST_SUITE_END();    // generate
