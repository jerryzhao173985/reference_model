
// Copyright (c) 2020, 2023-2024, ARM Limited.
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

#include "custom.h"
#include "attribute.h"

#include "tensor.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

OpCustom::OpCustom(SubgraphTraverser* sgt_, TosaAttributeBase* attribute_, uint64_t id_)
    : GraphNode(sgt_, Op_CUSTOM, id_)
{
    // Init Attribute
    if (auto p = dynamic_cast<TosaCustomAttribute*>(attribute_))
        attribute = new TosaCustomAttribute(p);
}

OpCustom::~OpCustom()
{}

int OpCustom::checkTensorAttributes()
{
    // Get the pointer to customOp library
    auto domain_name_vec   = attribute->domain_name();
    auto operator_name_vec = attribute->operator_name();
    std::string domain_name(domain_name_vec.begin(), domain_name_vec.end());
    std::string operator_name(operator_name_vec.begin(), operator_name_vec.end());

    auto getCustomNodeFunc = MasterRegistry::get_op(domain_name, operator_name);
    ERROR_IF(getCustomNodeFunc == nullptr, "Can't find the custom shared library: %s::%s is not registered.",
             domain_name.c_str(), operator_name.c_str());
    this->custom_op_ptr = getCustomNodeFunc();

    return 0;
}

int OpCustom::eval()
{
    auto inputs        = getInputs();
    int32_t num_inputs = inputs.size();
    auto tosa_level    = g_func_config.tosa_level;
    LEVEL_CHECK(num_inputs <= tosa_level.MAX_TENSOR_LIST_SIZE,
                "num_inputs should be smaller than or equal to MAX_TENSOR_LIST_SIZE");

    auto implementation_attrs_vec = attribute->implementation_attrs();
    std::string implementation_attrs(implementation_attrs_vec.begin(), implementation_attrs_vec.end());
    custom_op_ptr->eval(inputs, getOutputs(), implementation_attrs);

    return GraphNode::eval();
}
