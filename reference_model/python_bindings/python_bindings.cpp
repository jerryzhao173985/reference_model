
// Copyright (c) 2024-2025, ARM Limited.
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

#include "model_runner.h"
#include <climits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

size_t get_itemsize_bits(tosa::TosaSerializationTensor* tens);
size_t get_num_elements(tosa::TosaSerializationTensor* tens);
py::array view_reshape(py::array_t<uint8_t> bytes, tosa::TosaSerializationTensor* ser_tensor);

struct IModelRunnerPyWrapper
{
    TosaReference::IModelRunner runner;
    tosa::TosaSerializationHandler handler;
    func_debug_t _func_debug;
    func_config_t _func_config;

    IModelRunnerPyWrapper(func_debug_t func_debug, func_config_t func_config)
    {
        _func_debug  = func_debug;
        _func_config = func_config;
        runner.setFuncDebug(func_debug);
        runner.setFuncConfig(func_config);
    }

    GraphStatus initialize(std::string tosa_binary)
    {
        tosa::tosa_err_t error = handler.LoadFileTosaFlatbuffer(tosa_binary.data(), tosa_binary.size());
        if (error != tosa::TOSA_OK)
        {
            WARNING("An error occurred while loading the TOSA model.");
            return GraphStatus::TOSA_ERROR;
        }

        GraphStatus status = runner.initialize(handler);
        if (status != GraphStatus::TOSA_VALID)
        {
            WARNING("An error occurred while initializing the TOSA graph.");
        }
        return status;
    }

    int setInput(std::string input_name, py::array input)
    {
        py::buffer_info info;
        if (input.flags() & py::array::c_style)
        {
            info = input.request();
        }
        else
        {
            // If the array isn't in row-major order, we run the Python .copy() method
            py::array copied_input = input.attr("copy")();
            info                   = copied_input.request();
        }
        return runner.setInput(input_name, static_cast<uint8_t*>(info.ptr), info.size * info.itemsize);
    }

    int setInputs(std::vector<py::array> inputs)
    {
        std::vector<std::string> input_names = handler.GetMainRegion()->GetBlocks()[0]->GetInputs();
        if (inputs.size() != input_names.size())
        {
            WARNING("Incorrect number of input arrays provided.");
            return 1;
        }
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            int status = setInput(input_names[i], inputs[i]);
            if (status != 0)
            {
                return status;
            }
        }
        return 0;
    }

    GraphStatus run()
    {
        GraphStatus status = runner.run();
        if (status != GraphStatus::TOSA_VALID)
        {
            WARNING("An error occurred when running the model.");
        }

        // Manually flushing the debug output, since log files end up incomplete
        // otherwise when multiprocessing
        fflush(_func_debug.func_debug_file);

        return status;
    }

    std::vector<py::array> getOutputs()
    {
        std::vector<std::string> output_names = handler.GetMainRegion()->GetBlocks()[0]->GetOutputs();
        std::vector<py::array> outputs;
        for (std::string output_name : output_names)
        {
            outputs.push_back(getOutputBytes(output_name));
        }
        return outputs;
    }

    py::array getOutputBytes(std::string output_name)
    {
        tosa::TosaSerializationTensor* ser_tensor =
            handler.GetMainRegion()->GetBlocks()[0]->GetTensorByName(output_name);
        if (ser_tensor == nullptr)
        {
            return {};
        }

        size_t num_elements  = get_num_elements(ser_tensor);
        size_t itemsize_bits = get_itemsize_bits(ser_tensor);

        // Rounding up while dividing by 8 in case of int4 padding etc.
        size_t size_bytes = (num_elements * itemsize_bits + 8 - 1) / 8;
        uint8_t buffer[size_bytes];
        int status = runner.getOutput(output_name, buffer, size_bytes);
        if (status != 0)
        {
            return {};
        }

        py::array_t<uint8_t> bytes(size_bytes, buffer);

        return view_reshape(bytes, ser_tensor);
    }
};

std::pair<std::vector<py::array>, GraphStatus> run_model(string tosa_binary,
                                                         std::vector<py::array> inputs,
                                                         uint64_t inst_id,
                                                         std::optional<string> debug_filename,
                                                         bool is_unbuffered,
                                                         std::optional<string> debug_mode,
                                                         string verbosity,
                                                         uint32_t dump_intermediates,
                                                         uint32_t initialize_variable_tensor_from_numpy,
                                                         string fp_format,
                                                         uint32_t precise_mode,
                                                         bool bounds_mode,
                                                         string tosa_level)
{

    // Setting debug/config parameters
    func_debug_t func_debug;
    func_config_t func_config;

    func_debug.init_debug(inst_id);

    func_debug.set_output_unbuffered(is_unbuffered);
    if (debug_filename)
        func_debug.set_file(*debug_filename);
    if (debug_mode)
        func_debug.set_mask(*debug_mode);
    func_debug.set_verbosity(verbosity);

    func_config.dump_intermediates                    = dump_intermediates;
    func_config.initialize_variable_tensor_from_numpy = initialize_variable_tensor_from_numpy;
    func_config.fp_format                             = fp_format;
    func_config.precise_mode                          = precise_mode;
    func_config.bounds_mode                           = bounds_mode;

    if (tosa_level == "NONE")
        func_config.tosa_level = func_config_t::NONE;
    else if (tosa_level == "EIGHTK")
        func_config.tosa_level = func_config_t::EIGHTK;
    else
        throw std::invalid_argument("tosa_level must be NONE or EIGHTK");

    // Initializing and running model
    auto model = IModelRunnerPyWrapper(func_debug, func_config);

    if (model.initialize(tosa_binary) != GraphStatus::TOSA_VALID)
    {
        throw std::invalid_argument("Invalid TOSA graph");
    };

    if (model.setInputs(inputs) != 0)
    {
        throw std::invalid_argument("Invalid inputs.");
    }

    auto status = model.run();
    return { model.getOutputs(), status };
}

PYBIND11_MODULE(tosa_reference_model, m)
{
    using namespace TosaReference;
    using namespace pybind11::literals;

    std::stringstream ss;

    std::string docstring = R"(Runs reference model on serialized TOSA graph with given inputs.

Parameters
----------
tosa_binary : str
    TOSA graph serialized with flatbuffers using the TOSA serialization_lib
inputs : list[numpy.ndarray]
    List of inputs. Inputs are assigned by their order in the serialized graph.
inst_id : int, default=0
    The instance id for multiple model instances
debug_filename : str, optional
    Filename for debug output. If not set then debug is written to stderr.
is_unbuffered : bool, default=False
    Whether debug output is unbuffered
debug_mode : str, optional
    )" + func_debug_t().get_debug_mask_help_string() +
                            R"(. If not set, no debug info will be printed.
verbosity : str, default='NONE'
    )" + func_debug_t().get_debug_verbosity_help_string() +
                            R"(
dump_intermediates : int, default=0
    Dump intermediate tensors (0/1)
initialize_variable_tensor_from_numpy : int, default=0
    Initialize variable tensors from flatbuffer (0, default) or numpy (1)
fp_format : str, default='0.5'
    Floating-point number dump format string (printf-style format, e.g. 0.5)
precise_mode : int, default=0
    Calculate floating point operations in FP64 (0/1)
bounds_mode : bool, default=False
    Take absolute values of operands
tosa_level : str, default='NONE'
    TOSA level (NONE, EIGHTK)

Returns
-------
outputs : list[numpy.ndarray]
    List of outputs as arrays of bytes.
status : tosa_reference_model.GraphStatus
    Status of graph after running.

Raises
------
ValueError
    If a parameter is invalid, or an error occurs while setting inputs.
)";

    py::enum_<GraphStatus>(m, "GraphStatus")
        .value("TOSA_VALID", GraphStatus::TOSA_VALID)
        .value("TOSA_UNPREDICTABLE", GraphStatus::TOSA_UNPREDICTABLE)
        .value("TOSA_ERROR", GraphStatus::TOSA_ERROR);

    m.def("run", &run_model, docstring.c_str(), "tosa_binary"_a, "inputs"_a, py::kw_only(), "inst_id"_a = 0,
          "debug_filename"_a = nullptr, "is_unbuffered"_a = false, "debug_mode"_a = nullptr, "verbosity"_a = "NONE",
          "dump_intermediates"_a = 0, "initialize_variable_tensor_from_numpy"_a = 0, "fp_format"_a = "0.5",
          "precise_mode"_a = 0, "bounds_mode"_a = false, "tosa_level"_a = "NONE");
}

// Helpers

size_t get_itemsize_bits(tosa::TosaSerializationTensor* tens)
{
    // The types supported by ModelRunnerImpl::getOutput(). In bits
    // instead of bytes, to support INT4 in the future.
    switch (tens->GetDtype())
    {
        case tosa::DType_FP16:
            return 8 * sizeof(ct::float16);
        case tosa::DType_FP32:
            return 8 * sizeof(float);
        case tosa::DType_BF16:
            return 8 * sizeof(bf16);
        case tosa::DType_FP8E4M3:
            return 8 * sizeof(fp8e4m3);
        case tosa::DType_FP8E5M2:
            return 8 * sizeof(fp8e5m2);
        case tosa::DType_BOOL:
            return 8 * sizeof(bool);
        case tosa::DType_INT8:
            return 8 * sizeof(int8_t);
        case tosa::DType_INT16:
            return 8 * sizeof(int16_t);
        case tosa::DType_INT32:
            return 8 * sizeof(int32_t);
        default:
            return 0;
    }
}

size_t get_num_elements(tosa::TosaSerializationTensor* tens)
{
    size_t num_elements = 1;
    for (auto dim : tens->GetShape())
        num_elements *= dim;
    return num_elements;
}

py::array view_reshape(py::array_t<uint8_t> bytes, tosa::TosaSerializationTensor* ser_tensor)
{
    // Trying to set the array's type and shape to what is specified by the input graph.
    // In the default case, a raw flattened array of bytes is returned.

    switch (ser_tensor->GetDtype())
    {
        case tosa::DType_FP16:
            return bytes.view("<f2").reshape(ser_tensor->GetShape());
        case tosa::DType_FP32:
            return bytes.view("<f4").reshape(ser_tensor->GetShape());
        case tosa::DType_BOOL:
            return bytes.view("bool").reshape(ser_tensor->GetShape());
        case tosa::DType_INT8:
            return bytes.view("<i1").reshape(ser_tensor->GetShape());
        case tosa::DType_INT16:
            return bytes.view("<i2").reshape(ser_tensor->GetShape());
        case tosa::DType_INT32:
            return bytes.view("<i4").reshape(ser_tensor->GetShape());
        default:
            break;
    }

    switch (get_itemsize_bits(ser_tensor))
    {
        case 8:
            return bytes.view("<V1").reshape(ser_tensor->GetShape());
        case 16:
            return bytes.view("<V2").reshape(ser_tensor->GetShape());
        case 32:
            return bytes.view("<V4").reshape(ser_tensor->GetShape());
        case 64:
            return bytes.view("<V8").reshape(ser_tensor->GetShape());
        default:
            return bytes;
    }
}
