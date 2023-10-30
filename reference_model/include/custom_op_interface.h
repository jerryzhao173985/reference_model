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

#ifndef CUSTOMOPINTERFACE_H
#define CUSTOMOPINTERFACE_H

#include "tensor.h"
#include <vector>

using namespace tosa;

namespace TosaReference
{
class CustomOpInterface
{
public:
    CustomOpInterface()                                       = default;
    virtual std::string getDomainName() const                 = 0;
    virtual std::string getOperatorName() const               = 0;
    virtual int eval(std::vector<TosaReference::Tensor*>& input_tensors,
                     std::vector<TosaReference::Tensor*>& output_tensors,
                     const std::string& implementation_attrs) = 0;
    virtual std::string getVersion() const                    = 0;
};
}    // namespace TosaReference

#endif
