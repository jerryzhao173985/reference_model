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

#ifndef CUSTOMREGISTRY_H
#define CUSTOMREGISTRY_H

#include "custom_op_interface.h"
#include <unordered_map>

using namespace tosa;

namespace TosaReference
{

typedef CustomOpInterface* (*op_creation_function_t)();
typedef int (*registration_callback_t)(const std::string& domain_name,
                                       const std::string& operator_name,
                                       const op_creation_function_t& op_creation_function);

class MasterRegistry
{
public:
    static int register_function(const std::string& domain_name,
                                 const std::string& operator_name,
                                 const op_creation_function_t& op_creation_function)
    {
        std::string unique_id    = domain_name + "::" + operator_name;
        MasterRegistry& instance = get_instance();
        if (instance.op_creation_map.find(unique_id) != instance.op_creation_map.end())
        {
            std::cout << std::endl;
            printf("domain_name: %s and operator_name: %s pair has already been registered", domain_name.c_str(),
                   operator_name.c_str());
            return 1;
        }
        instance.op_creation_map[unique_id] = op_creation_function;
        return 0;
    }

    static MasterRegistry& get_instance()
    {
        static MasterRegistry instance;
        return instance;
    }

    MasterRegistry(const MasterRegistry&) = delete;
    void operator=(const MasterRegistry&) = delete;

    std::unordered_map<std::string, op_creation_function_t> get_ops() const
    {
        return op_creation_map;
    }

    static op_creation_function_t get_op(const std::string& domain_name, const std::string& operator_name)
    {
        std::string unique_id    = domain_name + "::" + operator_name;
        MasterRegistry& instance = get_instance();
        auto all_ops_map         = instance.get_ops();
        if (all_ops_map.find(unique_id) == all_ops_map.end())
        {
            return nullptr;
        }
        else
        {
            op_creation_function_t& op_creation_function = all_ops_map[unique_id];
            return op_creation_function;
        }
    }

private:
    MasterRegistry() = default;
    std::unordered_map<std::string, op_creation_function_t> op_creation_map;
};
}    // namespace TosaReference

#endif
