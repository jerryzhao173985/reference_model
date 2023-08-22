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
#include "verify.h"

#include <doctest.h>

#include <array>
#include <string>
#include <vector>

namespace
{

class TosaTensor
{
public:
    TosaTensor(std::string name, tosa_datatype_t dataType, std::vector<int32_t> shape)
        : _name(std::move(name))
        , _shape(std::move(shape))
    {
        _tensor.name      = _name.c_str();
        _tensor.data_type = dataType;
        _tensor.num_dims  = _shape.size();
        _tensor.shape     = _shape.data();
    };

    const tosa_tensor_t* cTensor() const
    {
        return &_tensor;
    }

private:
    std::string _name;
    std::vector<int32_t> _shape;
    tosa_tensor_t _tensor;
};

}    // namespace

TEST_SUITE_BEGIN("verify");

TEST_CASE("negative - api")
{
    std::string json_cfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "DOT_PRODUCT",
                "dot_product_info" : {
                    "data_type": "FP32",
                    "s": 2,
                    "ks": 9
                }
            }
        }
    })";

    SUBCASE("invalid json")
    {
        std::string invalid_json_cfg = R"({
            "tensors" : {
                "out1" : {
                    "mode": DOT_PRODUCT,
                },
            }
        })";

        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), invalid_json_cfg.c_str()));
    }
    SUBCASE("mismatching dimensions")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), json_cfg.c_str()));
    }
    SUBCASE("mismatching shapes")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 4, 4, 4 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), json_cfg.c_str()));
    }
    SUBCASE("mismatching data types")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp16_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), json_cfg.c_str()));
    }
    SUBCASE("missing tensor data")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), json_cfg.c_str()));
    }
}

TEST_SUITE_END();    // verify