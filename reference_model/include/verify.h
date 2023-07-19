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
//===----------------------------------------------------------------------===//
//
// Verification functionality as per TOSA Specification
// Output Verification : Section 1.8.2
//
//===----------------------------------------------------------------------===//

#include <cstddef>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    // Check result
    //
    // Error is valid only and only if is_valid is true
    struct CheckResult
    {
        bool is_valid;
        double error;
    };

    /// Validate and calculate tensor element error when using an fp32 accumulator
    ///
    /// \param ref Tensor element calculated using fp64
    /// \param bnd Tensor element calculated using fp64 on abs(input, weights)
    /// \param imp Tensor element calculated through the implementation
    /// \param KS The kernel size
    ///
    /// \return Output error
    CheckResult tosa_validate_element_accfp32(double ref, double bnd, float imp, size_t KS);

    /// Validate the accumulated output error
    ///
    /// \param err_sum Sum of error of all the tensor elements within a tensor
    /// \param err_sum_sq Sum of error squares of all the tensor elements within a tensor
    /// \param T Number of output (dot-product) elements
    /// \param KS The kernel size
    /// \param S Test set used as a input/weight generator
    ///
    /// \return True if the error is within margin else false
    bool tosa_validate_output_error(double err_sum, double err_sum_sq, size_t T, size_t KS, int S);

    /// Validate error of whole vector of output data
    ///
    /// \param ref Output elements calculated using fp64
    /// \param bnd Output elements calculated using fp64 on abs(input, weights)
    /// \param imp Output elements calculated using the implementation
    /// \param T Number of elements in outputs (need to match)
    /// \param KS The kernel size
    /// \param S Test set used as a input/weight generator
    ///
    /// \return True if the error is within margin else false
    bool tosa_validate_data_fp32(const double* ref, const double* bnd, const float* imp, size_t T, size_t KS, int S);

#ifdef __cplusplus
}
#endif /* __cplusplus */