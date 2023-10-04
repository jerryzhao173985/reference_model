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
#include <string>
#include <vector>

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
                    "ks": 10,
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
                    "ks": 10,
                    "acc_type": "FP32"
                }
            }
        }
    })";

    const std::string tosaName = "in1";
    const size_t tosaElements  = 4 * 8 * 8;
    const size_t tosaSize      = tosaElements * 4;

    SUBCASE("matmul")
    {
        std::vector<float> buffer(tosaElements);
        REQUIRE(tgd_generate_data(json_cfg.c_str(), tosaName.c_str(), (void*)buffer.data(), tosaSize));
        REQUIRE(buffer[0] == (float)-0.950864);
        REQUIRE(buffer[1] == 0.f);
    }
}

TEST_SUITE_END();    // generate