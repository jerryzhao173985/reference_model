// Copyright (c) 2023, 2025 ARM Limited.
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

#include "custom_op_interface.h"
#include "custom_registry.h"
#include <vector>

#ifdef _MSC_VER
#define TOSA_EXPORT __declspec(dllexport)
#else
#define TOSA_EXPORT
#endif

using namespace tosa;

namespace TosaReference
{
class CustomOpExample : public CustomOpInterface
{
public:
    CustomOpExample() = default;
    CustomOpExample(std::string& domain_name, std::string& operator_name, std::string& version)
        : _domain_name(domain_name)
        , _operator_name(operator_name)
        , _version(version)
    {}
    int eval(std::vector<TosaReference::Tensor*>& input_tensors,
             std::vector<TosaReference::Tensor*>& output_tensors,
             const std::string& implementation_attrs) override
    {
        auto input_tensor_ptr  = input_tensors[0];
        auto output_tensor_ptr = output_tensors[0];

        // down_cast to EigenTensor
        using TIn  = Eigen::Tensor<float, 1>;
        using TOut = Eigen::Tensor<float, 1>;

        auto eigenInputTensor  = reinterpret_cast<TosaReference::TensorTemplate<TIn>*>(input_tensor_ptr);
        auto eigenOutputTensor = reinterpret_cast<TosaReference::TensorTemplate<TIn>*>(output_tensor_ptr);

        // Assign the input to output as an example
        // This is plug-in implementation specific
        auto fcn                       = [](float a) -> float { return a; };
        eigenOutputTensor->getTensor() = eigenInputTensor->getTensor().unaryExpr(fcn);

        return 0;
    };

    std::string getDomainName() const override
    {
        return this->_domain_name;
    }

    std::string getOperatorName() const override
    {
        return this->_operator_name;
    }

    std::string getVersion() const override
    {
        return this->_version;
    }

    ~CustomOpExample(){};

private:
    std::string _domain_name;
    std::string _operator_name;
    std::string _version;
};

CustomOpInterface* customOpExample()
{
    std::string domain_name         = "ExampleDomain";
    std::string operator_name       = "ExampleOp";
    std::string version             = "1.0";
    CustomOpInterface* customOp_ptr = new CustomOpExample(domain_name, operator_name, version);

    return customOp_ptr;
}

extern "C" TOSA_EXPORT int getCustomOpCreationFuncs(registration_callback_t registration_func)
{
    std::string domain_name   = "ExampleDomain";
    std::string operator_name = "ExampleOp";
    return registration_func(domain_name, operator_name, &customOpExample);
}

}    // namespace TosaReference
