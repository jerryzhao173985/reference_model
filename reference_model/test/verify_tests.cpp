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
#include <numeric>
#include <vector>

TEST_SUITE("verify")
{
    TEST_CASE("check_element_accfp32")
    {
        const size_t KS = 27;

        // Negative (bnd == 0.0)
        REQUIRE_FALSE(tosa_validate_element_accfp32(0.0, 0.0, 1.0, KS).is_valid);
        REQUIRE_FALSE(tosa_validate_element_accfp32(1.0, 0.0, 0.0, KS).is_valid);
        // Negative (bnd > 0.0)
        REQUIRE_FALSE(tosa_validate_element_accfp32(5.0, 5.0, 5.1, KS).is_valid);

        // Positive (bnd == 0.0 && ref == 0.0 && imp == 0.0)
        REQUIRE(tosa_validate_element_accfp32(0.0, 0.0, 0.0, KS).is_valid);
        REQUIRE(tosa_validate_element_accfp32(0.0, 0.0, 0.0, KS).error == 0.0);

        // Positive (bnd > 0.0)
        REQUIRE(tosa_validate_element_accfp32(4.0, 4.0, 4.0, KS).error == 0.0);
        REQUIRE(tosa_validate_element_accfp32(4.0, 4.0, 4.0, KS).error == 0.0);
        REQUIRE(tosa_validate_element_accfp32(4.0, 4.0, 4.0, KS).error == 0.0);
    }
    TEST_CASE("check_output_error")
    {
        const size_t KS = 27;
        const size_t T  = 1024;

        // Negative (S!=1 && S!=2 && (abs(err_sum) > 2*sqrt(KS*T)))
        REQUIRE_FALSE(tosa_validate_output_error(1560, 112000, KS, T, 0));
        // Negative (err_sum_sq > 0.4*KS*T))
        REQUIRE_FALSE(tosa_validate_output_error(1560, 112000, KS, T, 1));
        // Positive
        REQUIRE(tosa_validate_output_error(10, 254, KS, T, 0));
        REQUIRE(tosa_validate_output_error(10, 254, KS, T, 1));
    }
}
