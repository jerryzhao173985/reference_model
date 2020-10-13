
// Copyright (c) 2020, ARM Limited.
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

#include "tosa_serialization_handler.h"

#include <iostream>
using namespace tosa;

TosaSerializationTensor::TosaSerializationTensor(const flatbuffers::String* name,
                                                 const flatbuffers::Vector<uint32_t>& usage,
                                                 const flatbuffers::Vector<int32_t>& shape,
                                                 DType dtype,
                                                 const flatbuffers::Vector<uint32_t>& format,
                                                 const flatbuffers::String* npy_filename)
{
    _dtype = dtype;

    _usage = new std::vector<Usage>(usage.size());
    for (uint32_t us : usage)
    {
        _usage->push_back((Usage)us);
    }
    assert(_usage);

    _format = new std::vector<Format>(format.size());
    for (uint32_t fm : format)
    {
        _format->push_back((Format)fm);
    }
    assert(_format);

    _shape = new std::vector<int32_t>(shape.begin(), shape.end());

    _shape = new std::vector<int32_t>(shape.begin(), shape.end());
    assert(_shape);

    assert(name);
    _name = new std::string(name->str());
    assert(_name);

    if (npy_filename)
    {
        _npy_filename = new std::string(npy_filename->str());
        assert(_npy_filename);
    }
    else
    {
        _npy_filename = nullptr;
    }
}

TosaSerializationTensor::TosaSerializationTensor(std::string name,
                                                 const std::vector<Usage>& usage,
                                                 const std::vector<int32_t>& shape,
                                                 DType dtype,
                                                 const std::vector<Format>& format,
                                                 const std::string* npy_filename)
{

    _dtype = dtype;

    _usage = new std::vector<Usage>(usage);
    assert(_usage);

    _format = new std::vector<Format>(format);
    assert(_format);

    _shape = new std::vector<int32_t>(shape);
    assert(_shape);

    _name = new std::string(name);
    assert(_name);

    if (npy_filename)
    {
        _npy_filename = new std::string(*npy_filename);
        assert(_npy_filename);
    }
    else
    {
        _npy_filename = nullptr;
    }
}

TosaSerializationTensor::TosaSerializationTensor()
{
    _dtype = DType_UNKNOWN;

    _usage  = new std::vector<Usage>();
    _format = new std::vector<Format>();
    _shape  = new std::vector<int32_t>();
    _name   = new std::string("UNKNOWN");
    assert(_usage && _format && _shape && _name);

    _npy_filename = nullptr;
}

TosaSerializationTensor::TosaSerializationTensor(const TosaSerializationTensor& rhs)
{
    _dtype = rhs._dtype;

    assert(rhs._usage);
    _usage = new std::vector<Usage>(*rhs._usage);
    assert(_usage);

    assert(rhs._format);
    _format = new std::vector<Format>(*rhs._format);
    assert(_format);

    assert(rhs._shape);
    _shape = new std::vector<int32_t>(*rhs._shape);
    assert(_shape);

    assert(rhs._name);
    _name = new std::string(*rhs._name);
    assert(_name);

    if (rhs._npy_filename)
    {
        _npy_filename = new std::string(*rhs._npy_filename);
        assert(_npy_filename);
    }
    else
    {
        _npy_filename = nullptr;
    }
}

TosaSerializationTensor& TosaSerializationTensor::operator=(const TosaSerializationTensor& rhs)
{
    _dtype = rhs._dtype;

    delete _usage;
    assert(rhs._usage);
    _usage = new std::vector<Usage>(*rhs._usage);
    assert(_usage);

    delete _format;
    assert(rhs._format);
    _format = new std::vector<Format>(*rhs._format);
    assert(_format);

    delete _shape;
    assert(rhs._shape);
    _shape = new std::vector<int32_t>(*rhs._shape);
    assert(_shape);

    delete _name;
    assert(rhs._name);
    _name = new std::string(*rhs._name);
    assert(_name);

    if (_npy_filename)
        delete _npy_filename;

    if (rhs._npy_filename)
    {
        _npy_filename = new std::string(*rhs._npy_filename);
    }
    else
    {
        _npy_filename = nullptr;
    }
    return *this;
}

TosaSerializationTensor::TosaSerializationTensor(TosaSerializationTensor&& rhs)
{
    _dtype = rhs._dtype;
    std::swap(_format, rhs._format);
    std::swap(_usage, rhs._usage);
    std::swap(_shape, rhs._shape);
    std::swap(_name, rhs._name);
    std::swap(_npy_filename, rhs._npy_filename);
}

TosaSerializationTensor& TosaSerializationTensor::operator=(TosaSerializationTensor&& rhs)
{
    _dtype = rhs._dtype;
    std::swap(_format, rhs._format);
    std::swap(_usage, rhs._usage);
    std::swap(_shape, rhs._shape);
    std::swap(_name, rhs._name);
    std::swap(_npy_filename, rhs._npy_filename);
    return *this;
}

TosaSerializationTensor::~TosaSerializationTensor()
{
    delete _usage;
    delete _format;
    delete _shape;
    delete _name;
    if (_npy_filename)
        delete _npy_filename;
}

TosaSerializationOperator::TosaSerializationOperator(Op op,
                                                     Attribute attribute_type,
                                                     const TosaAttributeBase* attribute,
                                                     QuantInfo qinfo_type,
                                                     const TosaQuantInfoBase* qinfo,
                                                     std::vector<std::string> input_tensor_names,
                                                     std::vector<std::string> output_tensor_names)
{
    _op             = op;
    _attribute_type = attribute_type;

    switch (attribute_type)
    {
        case Attribute_NONE:
            _attribute = new TosaNoneAttribute();
            break;
#define DEF_ATTRIBUTE(NAME, ...)                                                                                       \
    case Attribute_##NAME##Attribute:                                                                                  \
        _attribute = new Tosa##NAME##Attribute(attribute);                                                             \
        break;
#include "attribute.def"
#undef DEF_ATTRIBUTE
        default:
            printf("TosaSerializationOperator::TosaSerializationOperator(): Attribute %s not implemented yet\n",
                   EnumNamesAttribute()[attribute_type]);
            assert(0);
    }

    _qinfo_type = qinfo_type;
    switch (qinfo_type)
    {
        case QuantInfo_NONE:
            _qinfo = new TosaNoneQuantInfo();
            break;
#define DEF_QUANTIZATION_INFO(NAME, ...)                                                                               \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        _qinfo = new Tosa##NAME##QuantInfo(qinfo);                                                                     \
        break;
#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
        default:
            printf("TosaSerializationOperator::TosaSerializationOperator(): QuantInfo %s not implemented yet\n",
                   EnumNamesQuantInfo()[qinfo_type]);
            assert(0);
    }

    assert(_attribute && _qinfo);

    _input_tensor_names  = new std::vector<std::string>(input_tensor_names);
    _output_tensor_names = new std::vector<std::string>(output_tensor_names);

    assert(_input_tensor_names && _output_tensor_names);

    _input_tensors  = new std::vector<TosaSerializationTensor*>();
    _output_tensors = new std::vector<TosaSerializationTensor*>();

    assert(_input_tensors && _output_tensors);
}

TosaSerializationOperator::~TosaSerializationOperator()
{
    delete _attribute;
    delete _qinfo;
    delete _input_tensor_names;
    delete _output_tensor_names;
    // TosaSerializationTensor should be free'd in TosaSerializationSerializationHandler destructor
    delete _input_tensors;
    delete _output_tensors;
}

TosaSerializationBasicBlock::TosaSerializationBasicBlock(std::string name,
                                                         std::vector<TosaSerializationOperator*> operators,
                                                         std::vector<TosaSerializationTensor*> tensors,
                                                         std::vector<std::string> inputs,
                                                         std::vector<std::string> outputs)
{

    _name = new std::string(name);
    assert(_name);

    _operators = new std::vector<TosaSerializationOperator*>(operators);
    assert(_operators);

    _tensors = new std::vector<TosaSerializationTensor*>(tensors);
    assert(_tensors);

    _inputs = new std::vector<std::string>(inputs);
    assert(_inputs);

    _outputs = new std::vector<std::string>(outputs);
    assert(_outputs);
}

TosaSerializationBasicBlock::~TosaSerializationBasicBlock()
{
    delete _name;

    // deallocate all operators
    for (auto op : GetOperators())
    {
        delete op;    // ~TosaSerializationOperator()
    }
    delete _operators;

    // deallocate all tensors
    for (auto ts : GetTensors())
    {
        delete ts;    // ~TosaSerializationTensor()
    }
    _tensors->clear();

    delete _inputs;
    delete _outputs;
}

TosaSerializationHandler::TosaSerializationHandler()
{
    _schemaLoaded = false;
    _builder      = new flatbuffers::FlatBufferBuilder();
    _parser       = new flatbuffers::Parser();
    _blocks       = new std::vector<TosaSerializationBasicBlock*>();

    assert(_builder && _parser && _blocks);

    SetTosaVersion();
}

TosaSerializationHandler::~TosaSerializationHandler()
{
    if (_version)
        delete _version;
    delete _builder;
    delete _parser;

    Clear();    // deallocate all basic blocks

    delete _blocks;
}

tosa_err_t TosaSerializationHandler::SetTosaVersion()
{
    // version is specified within .fbs
    // and it's encoded as defaulted value of CreateTosaVersion()
    // need to write out one object to read that value out
    // TODO: very costly now. is there any better way to encode constant in .fbs?
    auto fboffset_version    = CreateVersion(*_builder);
    auto fboffset_tosa_graph = CreateTosaGraphDirect(*_builder, fboffset_version, nullptr);
    _builder->Finish(fboffset_tosa_graph);
    std::string jsongen;
    uint8_t* buf         = _builder->GetBufferPointer();
    auto fb_tosa_graph   = GetTosaGraph(buf);
    auto fb_tosa_version = fb_tosa_graph->version();

    _version = new TosaVersion(fb_tosa_version->_major(), fb_tosa_version->_minor(), fb_tosa_version->_patch(),
                               fb_tosa_version->_experimental());

    assert(_version);
    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::LoadFileSchema(const char* schema_filename)
{
    std::string schema;
    bool ok;

    ok = flatbuffers::LoadFile(schema_filename, false, &schema);
    if (!ok)
    {
        printf("Error loading schema file: %s\n", schema_filename);
        return TOSA_FILE_ERROR;
    }

    ok = _parser->Parse(schema.c_str());
    if (!ok)
    {
        printf("Error parsing ISA schema file: %s\n", schema_filename);
        return TOSA_FILE_ERROR;
    }
    _schemaLoaded = true;

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::LoadFileJson(const char* filename)
{
    std::string jsonfile;
    bool ok;
    tosa_err_t err;

    if (!_schemaLoaded)
    {
        return TOSA_SCHEMA_MISSING;
    }

    ok = flatbuffers::LoadFile(filename, false, &jsonfile);
    if (!ok)
    {
        printf("Error loading json file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    ok = _parser->Parse(jsonfile.c_str());
    if (!ok)
    {
        printf("Error parsing json file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    uint8_t* buf = _parser->builder_.GetBufferPointer();

    err = InitWithBuf(buf);
    if (err != TOSA_OK)
    {
        return err;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::SaveFileJson(const char* filename)
{
    std::string jsongen;
    tosa_err_t err;

    if (!_schemaLoaded)
    {
        return TOSA_SCHEMA_MISSING;
    }

    err = FreezeBuilder();
    if (err != TOSA_OK)
    {
        return err;
    }

    uint8_t* buf = _builder->GetBufferPointer();

    if (!GenerateText(*_parser, buf, &jsongen))
    {
        printf("Couldn't serialize parsed data to JSON!\n");
        return TOSA_FILE_ERROR;
    }

    FILE* file = fopen(filename, "wb");

    if (!file)
    {
        printf("Couldn't open output file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    if (fwrite(jsongen.c_str(), sizeof(char), jsongen.size(), file) != jsongen.size())
    {
        printf("Error writing to json output file: %s\n", filename);
        fclose(file);
        return TOSA_FILE_ERROR;
    }

    if (file)
        fclose(file);

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::LoadFileTosaFlatbuffer(const char* filename)
{
    std::string read_buffer;
    tosa_err_t err;
    uint8_t* buf;
    bool ok;

    ok = flatbuffers::LoadFile(filename, false, &read_buffer);
    if (!ok)
    {
        printf("Error loading flatbuffer file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    buf = (uint8_t*)read_buffer.data();

    err = InitWithBuf(buf);
    if (err != TOSA_OK)
    {
        return err;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::SaveFileTosaFlatbuffer(const char* filename)
{
    tosa_err_t err;

    err = FreezeBuilder();
    if (err != TOSA_OK)
    {
        return err;
    }

    uint8_t* buf = _builder->GetBufferPointer();

    bool ok = flatbuffers::SaveFile(filename, (const char*)buf, _builder->GetSize(), false);
    if (!ok)
    {
        printf("Error saving floatbuffer file: %s\n", filename);
        return TOSA_FILE_ERROR;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::Clear()
{
    // deallocate all basic blocks
    for (auto bb : GetBlocks())
    {
        delete bb;
    }
    _blocks->clear();

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::CheckTosaVersion(const TosaVersion& read_version)
{
    if ((*_version) != read_version)
    {
        printf("WARNING: read tosa version: %s != schema tosa version %s\n", read_version.to_string().c_str(),
               this->_version->to_string().c_str());
        return TOSA_VERSION_MISMATCH;
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::InitWithBuf(const uint8_t* buf)
{
    auto fb_tosa_graph   = GetTosaGraph(buf);
    auto fb_tosa_version = fb_tosa_graph->version();
    auto fb_tosa_blocks  = fb_tosa_graph->blocks();

    std::vector<std::string> operator_inputs_container;
    std::vector<std::string> operator_outputs_container;

    std::vector<TosaSerializationOperator*> block_operators_container;
    std::vector<TosaSerializationTensor*> block_tensors_container;
    std::vector<std::string> block_inputs_container;
    std::vector<std::string> block_outputs_container;

    TosaAttributeBase* typed_attribute      = NULL;
    TosaQuantInfoBase* typed_qinfo          = NULL;
    TosaSerializationOperator* new_operator = NULL;
    TosaSerializationBasicBlock* new_block  = NULL;
    TosaSerializationTensor* new_tensor     = NULL;

    // erase container
    Clear();

    TosaVersion read_version(fb_tosa_version->_major(), fb_tosa_version->_minor(), fb_tosa_version->_patch(),
                             fb_tosa_version->_experimental());
    tosa_err_t err = CheckTosaVersion(read_version);

    if (err != TOSA_OK)
        return err;

    for (size_t i = 0; i < fb_tosa_blocks->size(); i++)
    {
        auto curr_block = fb_tosa_blocks->Get(i);

        auto block_name = curr_block->name()->str();

        auto fb_tosa_operators = curr_block->operators();
        block_operators_container.clear();
        for (size_t j = 0; j < fb_tosa_operators->size(); j++)
        {
            auto curr_operator = fb_tosa_operators->Get(j);

            auto operator_op         = curr_operator->op();
            auto attribute_type      = curr_operator->attribute_type();
            auto attribute           = curr_operator->attribute();
            auto operator_qinfo_type = curr_operator->quant_info_type();
            auto operator_qinfo      = curr_operator->quant_info();

            // input tensors
            auto operator_inputs = curr_operator->inputs();
            operator_inputs_container.clear();
            if (operator_inputs)
            {
                for (size_t k = 0; k < operator_inputs->size(); k++)
                {
                    auto curr_input = operator_inputs->Get(k);
                    operator_inputs_container.push_back(curr_input->str());
                }
            }

            // output tensors
            auto operator_outputs = curr_operator->outputs();
            operator_outputs_container.clear();
            if (operator_outputs)
            {
                for (size_t k = 0; k < operator_outputs->size(); k++)
                {
                    auto curr_output = operator_outputs->Get(k);
                    operator_outputs_container.push_back(curr_output->str());
                }
            }

            switch (attribute_type)
            {
                case Attribute_NONE:
                    typed_attribute = new TosaNoneAttribute();
                    break;
#define DEF_ATTRIBUTE(NAME, ...)                                                                                       \
    case Attribute_##NAME##Attribute:                                                                                  \
        typed_attribute = new Tosa##NAME##Attribute(attribute);                                                        \
        break;
#include "attribute.def"
#undef DEF_ATTRIBUTE
                default:
                    printf("TosaSerializationHandler::InitWithBuf(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            switch (operator_qinfo_type)
            {
                case QuantInfo_NONE:
                    typed_qinfo = new TosaNoneQuantInfo();
                    break;
#define DEF_QUANTIZATION_INFO(NAME, ...)                                                                               \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        typed_qinfo = new Tosa##NAME##QuantInfo(operator_qinfo);                                                       \
        break;

#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
                default:
                    printf("TosaSerializationHandler::InitWithBuf(): QuantInfo %s not implemented yet\n",
                           EnumNamesQuantInfo()[operator_qinfo_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            new_operator =
                new TosaSerializationOperator(operator_op, attribute_type, typed_attribute, operator_qinfo_type,
                                              typed_qinfo, operator_inputs_container, operator_outputs_container);
            if (new_operator)
            {
                block_operators_container.push_back(new_operator);
            }
            else
            {
                return TOSA_MEMORY_ERROR;
            }

            if (typed_attribute)
                delete typed_attribute;
            if (typed_qinfo)
                delete typed_qinfo;
        }

        auto fb_tosa_tensors = curr_block->tensors();
        block_tensors_container.clear();
        for (size_t j = 0; j < fb_tosa_tensors->size(); j++)
        {
            auto curr_tensor = fb_tosa_tensors->Get(j);

            auto tensor_name         = curr_tensor->name();
            auto tensor_usage        = curr_tensor->usage();
            auto tensor_shape        = curr_tensor->shape();
            auto tensor_type         = curr_tensor->type();
            auto tensor_format       = curr_tensor->format();
            auto tensor_npy_filename = curr_tensor->npy_filename();

            new_tensor = new TosaSerializationTensor(tensor_name, *tensor_usage, *tensor_shape, tensor_type,
                                                     *tensor_format, tensor_npy_filename);
            if (new_tensor)
            {
                block_tensors_container.push_back(new_tensor);
            }
            else
            {
                return TOSA_MEMORY_ERROR;
            }
        }

        auto block_inputs  = curr_block->inputs();
        auto block_outputs = curr_block->outputs();

        block_inputs_container.clear();
        block_outputs_container.clear();

        for (size_t j = 0; j < block_inputs->size(); j++)
        {
            auto curr_block_input = block_inputs->Get(j);
            block_inputs_container.push_back(curr_block_input->str());
        }
        for (size_t j = 0; j < block_outputs->size(); j++)
        {
            auto curr_block_output = block_outputs->Get(j);
            block_outputs_container.push_back(curr_block_output->str());
        }

        new_block = new TosaSerializationBasicBlock(block_name, block_operators_container, block_tensors_container,
                                                    block_inputs_container, block_outputs_container);
        if (new_block)
        {
            this->GetBlocks().push_back(new_block);
        }
        else
        {
            return TOSA_MEMORY_ERROR;
        }
    }

    return TOSA_OK;
}

tosa_err_t TosaSerializationHandler::FreezeBuilder()
{
    std::vector<flatbuffers::Offset<TosaBasicBlock>> fboffset_blocks;

    std::vector<flatbuffers::Offset<TosaOperator>> fboffset_block_operators;
    std::vector<flatbuffers::Offset<TosaTensor>> fboffset_block_tensors;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_block_inputs;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_block_outputs;

    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_operator_inputs;
    std::vector<flatbuffers::Offset<flatbuffers::String>> fboffset_operator_outputs;

    // translate TosaFlatbufferOperator to flatbuffers::Offset<TosaOperator>
    for (auto block : GetBlocks())
    {
        fboffset_block_operators.clear();
        fboffset_block_tensors.clear();
        fboffset_block_inputs.clear();
        fboffset_block_outputs.clear();

        auto block_name = _builder->CreateString(block->GetName().c_str());

        for (auto tensor_str : block->GetInputs())
        {
            auto tensor_name = _builder->CreateString(tensor_str.c_str());
            fboffset_block_inputs.push_back(tensor_name);
        }

        for (auto tensor_str : block->GetOutputs())
        {
            auto tensor_name = _builder->CreateString(tensor_str.c_str());
            fboffset_block_outputs.push_back(tensor_name);
        }

        auto fb_block_inputs  = _builder->CreateVector(fboffset_block_inputs);
        auto fb_block_outputs = _builder->CreateVector(fboffset_block_outputs);

        for (auto op : block->GetOperators())
        {
            fboffset_operator_inputs.clear();
            fboffset_operator_outputs.clear();

            auto operator_op    = op->GetOp();
            auto attribute_type = op->GetAttributeType();

            for (auto tensor_str : op->GetInputTensorNames())
            {
                auto tensor_name = _builder->CreateString(tensor_str.c_str());
                fboffset_operator_inputs.push_back(tensor_name);
            }

            for (auto tensor_str : op->GetOutputTensorNames())
            {
                auto tensor_name = _builder->CreateString(tensor_str.c_str());
                fboffset_operator_outputs.push_back(tensor_name);
            }

            auto fb_operator_inputs  = _builder->CreateVector(fboffset_operator_inputs);
            auto fb_operator_outputs = _builder->CreateVector(fboffset_operator_outputs);

            flatbuffers::Offset<void> fb_attribute;
            switch (attribute_type)
            {
                case Attribute_NONE:
                    fb_attribute = 0;
                    break;

#define DEF_ARGS_S_STR(NAME, V) , _builder->CreateString(reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V().c_str())
#define DEF_ARGS_S_DEFAULT(NAME, V) , reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V()

#define DEF_ARGS_S_int32_t(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_float(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_bool(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_ResizeMode(NAME, V) DEF_ARGS_S_DEFAULT(NAME, V)
#define DEF_ARGS_S_string(NAME, V) DEF_ARGS_S_STR(NAME, V)

#define DEF_ARGS_S(NAME, T, V) DEF_ARGS_S_##T(NAME, V)
#define DEF_ARGS_V(NAME, T, V) , _builder->CreateVector<T>(reinterpret_cast<Tosa##NAME*>(op->GetAttribute())->V())

#define DEF_ARGS_1(NAME, T0, F0, V0) DEF_ARGS_##F0(NAME, T0, V0)
#define DEF_ARGS_2(NAME, T0, F0, V0, T1, F1, V1) DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1)
#define DEF_ARGS_3(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2)                                                           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2)
#define DEF_ARGS_4(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3)                                               \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)
#define DEF_ARGS_5(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4)                                   \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4)
#define DEF_ARGS_6(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5)                       \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5)
#define DEF_ARGS_7(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6)           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)
#define DEF_ATTRIBUTE(NAME, NUM_ARGS, ...)                                                                             \
    case Attribute_##NAME##Attribute:                                                                                  \
        fb_attribute = Create##NAME##Attribute(*_builder DEF_ARGS_##NUM_ARGS(NAME##Attribute, __VA_ARGS__)).Union();   \
        break;

#include "attribute.def"
#undef DEF_ATTRIBUTE
#undef DEF_ARGS_1
#undef DEF_ARGS_2
#undef DEF_ARGS_3
#undef DEF_ARGS_4
#undef DEF_ARGS_5
#undef DEF_ARGS_6
#undef DEF_ARGS_7
#undef DEF_ARGS_S
#undef DEF_ARGS_V
#undef DEF_ARGS_S_int32_t
#undef DEF_ARGS_S_float
#undef DEF_ARGS_S_bool
#undef DEF_ARGS_S_ResizeMode
#undef DEF_ARGS_S_string
#undef DEF_ARGS_S_STR
#undef DEF_ARGS_S_DEFAULT
                default:
                    printf("TosaSerializationHandler::FreezeBuilder(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            auto qinfo_type = op->GetQInfoType();
            flatbuffers::Offset<void> fb_operator_qinfo;
            switch (qinfo_type)
            {
                case QuantInfo_NONE:
                    fb_operator_qinfo = 0;
                    break;
#define DEF_ARGS_S(NAME, T, V) , reinterpret_cast<Tosa##NAME*>(op->GetQInfo())->V()
#define DEF_ARGS_V(NAME, T, V) , _builder->CreateVector<T>(reinterpret_cast<Tosa##NAME*>(op->GetQInfo())->V())

#define DEF_ARGS_1(NAME, T0, F0, V0) DEF_ARGS_##F0(NAME, T0, V0)
#define DEF_ARGS_2(NAME, T0, F0, V0, T1, F1, V1) DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1)
#define DEF_ARGS_3(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2)                                                           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2)
#define DEF_ARGS_4(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3)                                               \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)
#define DEF_ARGS_5(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4)                                   \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4)
#define DEF_ARGS_6(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5)                       \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5)
#define DEF_ARGS_7(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6)           \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)
#define DEF_ARGS_8(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,   \
                   V7)                                                                                                 \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7)
#define DEF_ARGS_9(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,   \
                   V7, T8, F8, V8)                                                                                     \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7) DEF_ARGS_##F8(NAME, T8, V8)
#define DEF_ARGS_10(NAME, T0, F0, V0, T1, F1, V1, T2, F2, V2, T3, F3, V3, T4, F4, V4, T5, F5, V5, T6, F6, V6, T7, F7,  \
                    V7, T8, F8, V8, T9, F9, V9)                                                                        \
    DEF_ARGS_##F0(NAME, T0, V0) DEF_ARGS_##F1(NAME, T1, V1) DEF_ARGS_##F2(NAME, T2, V2) DEF_ARGS_##F3(NAME, T3, V3)    \
        DEF_ARGS_##F4(NAME, T4, V4) DEF_ARGS_##F5(NAME, T5, V5) DEF_ARGS_##F6(NAME, T6, V6)                            \
            DEF_ARGS_##F7(NAME, T7, V7) DEF_ARGS_##F8(NAME, T8, V8) DEF_ARGS_##F9(NAME, T9, V9)
#define DEF_QUANTIZATION_INFO(NAME, NUM_ARGS, ...)                                                                     \
    case QuantInfo_##NAME##QuantInfo:                                                                                  \
        fb_operator_qinfo =                                                                                            \
            Create##NAME##QuantInfo(*_builder DEF_ARGS_##NUM_ARGS(NAME##QuantInfo, __VA_ARGS__)).Union();              \
        break;

#include "quant_info.def"
#undef DEF_QUANTIZATION_INFO
#undef DEF_ARGS_1
#undef DEF_ARGS_2
#undef DEF_ARGS_3
#undef DEF_ARGS_4
#undef DEF_ARGS_5
#undef DEF_ARGS_6
#undef DEF_ARGS_7
#undef DEF_ARGS_8
#undef DEF_ARGS_9
#undef DEF_ARGS_10
#undef DEF_ARGS_S
#undef DEF_ARGS_V
                default:
                    printf("TosaSerializationHandler::FreezeBuilder(): Attribute %s not implemented yet\n",
                           EnumNamesAttribute()[attribute_type]);
                    return TOSA_INTERNAL_ERROR;
            }

            auto fboffset_operator =
                CreateTosaOperator(*_builder, operator_op, attribute_type, fb_attribute, fb_operator_inputs,
                                   fb_operator_outputs, qinfo_type, fb_operator_qinfo);
            fboffset_block_operators.push_back(fboffset_operator);
        }

        auto fb_block_operators = _builder->CreateVector(fboffset_block_operators);

        for (auto tensor : block->GetTensors())
        {

            auto tensor_name = _builder->CreateString(tensor->GetName().c_str());
            auto tensor_usage =
                _builder->CreateVector(std::vector<uint32_t>(tensor->GetUsage().begin(), tensor->GetUsage().end()));
            auto tensor_shape = _builder->CreateVector(tensor->GetShape());
            auto tensor_dtype = tensor->GetDtype();
            auto tensor_format =
                _builder->CreateVector(std::vector<uint32_t>(tensor->GetFormat().begin(), tensor->GetFormat().end()));
            flatbuffers::Offset<flatbuffers::String> tensor_npy_filename = 0;
            if (tensor->GetNpyFilePtr())
                tensor_npy_filename = _builder->CreateString(tensor->GetNpyFilePtr()->c_str());

            auto fboffset_tensor = CreateTosaTensor(*_builder, tensor_name, tensor_shape, tensor_dtype, tensor_usage,
                                                    tensor_format, tensor_npy_filename);
            fboffset_block_tensors.push_back(fboffset_tensor);
        }

        auto fb_block_tensors = _builder->CreateVector(fboffset_block_tensors);

        auto fboffset_block = CreateTosaBasicBlock(*_builder, block_name, fb_block_operators, fb_block_tensors,
                                                   fb_block_inputs, fb_block_outputs);
        fboffset_blocks.push_back(fboffset_block);
    }

    auto fb_blocks = _builder->CreateVector(fboffset_blocks);

    auto fb_version = CreateVersion(*_builder, GetTosaVersion()->_major, GetTosaVersion()->_minor,
                                    GetTosaVersion()->_patch, GetTosaVersion()->_experimental);

    auto fb_graph = CreateTosaGraph(*_builder, fb_version, fb_blocks);
    _builder->Finish(fb_graph);

    return TOSA_OK;
}

// Magic NUMPY header
static const char NUMPY_HEADER_STR[] = "\x93NUMPY\x1\x0\x76\x0{";
static const int NUMPY_HEADER_SZ     = 128;

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, bool* databuf)
{
    const char dtype_str[] = "'|b1'";
    FILE* infile           = nullptr;
    NPError rc             = NO_ERROR;

    assert(filename);
    assert(databuf);

    infile = fopen(filename, "rb");
    if (!infile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    rc = checkNpyHeader(infile, elems, dtype_str);
    if (rc != NO_ERROR)
    {
        goto done;
    }

    // Read in the data from numpy byte array to native bool
    // array format
    for (uint32_t i = 0; i < elems; i++)
    {
        int val = fgetc(infile);

        if (val == EOF)
        {
            rc = FILE_IO_ERROR;
            goto done;
        }

        databuf[i] = val;
    }

done:

    if (infile)
        fclose(infile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, int32_t* databuf)
{
    const char dtype_str[] = "'<i4'";
    FILE* infile           = nullptr;
    NPError rc             = NO_ERROR;

    assert(filename);
    assert(databuf);

    infile = fopen(filename, "rb");
    if (!infile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    rc = checkNpyHeader(infile, elems, dtype_str);
    if (rc != NO_ERROR)
    {
        goto done;
    }

    // Now we are at the beginning of the data
    // Parse based on the datatype and number of dimensions
    if (fread(databuf, sizeof(int32_t), elems, infile) != elems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (infile)
        fclose(infile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, int64_t* databuf)
{
    const char dtype_str[] = "'<i8'";
    FILE* infile           = nullptr;
    NPError rc             = NO_ERROR;

    assert(filename);
    assert(databuf);

    infile = fopen(filename, "rb");
    if (!infile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    rc = checkNpyHeader(infile, elems, dtype_str);
    if (rc != NO_ERROR)
    {
        goto done;
    }

    // Now we are at the beginning of the data
    // Parse based on the datatype and number of dimensions
    if (fread(databuf, sizeof(int64_t), elems, infile) != elems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (infile)
        fclose(infile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::readFromNpyFile(const char* filename, const uint32_t elems, float* databuf)
{
    const char dtype_str[] = "'<f4'";
    FILE* infile           = nullptr;
    NPError rc             = NO_ERROR;

    assert(filename);
    assert(databuf);

    infile = fopen(filename, "rb");
    if (!infile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    rc = checkNpyHeader(infile, elems, dtype_str);
    if (rc != NO_ERROR)
    {
        goto done;
    }

    // Now we are at the beginning of the data
    // Parse based on the datatype and number of dimensions
    if (fread(databuf, sizeof(float), elems, infile) != elems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (infile)
        fclose(infile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::checkNpyHeader(FILE* infile, const uint32_t elems, const char* dtype_str)
{
    char buf[NUMPY_HEADER_SZ + 1];
    char* ptr         = nullptr;
    NPError rc        = NO_ERROR;
    bool foundFormat  = false;
    bool foundOrder   = false;
    bool foundShape   = false;
    bool fortranOrder = false;
    std::vector<int> shape;
    uint32_t totalElems = 1;
    char* outer_end     = NULL;

    assert(infile);
    assert(elems > 0);

    if (fread(buf, NUMPY_HEADER_SZ, 1, infile) != 1)
    {
        rc = HEADER_PARSE_ERROR;
        goto done;
    }

    if (memcmp(buf, NUMPY_HEADER_STR, sizeof(NUMPY_HEADER_STR) - 1))
    {
        rc = HEADER_PARSE_ERROR;
        goto done;
    }

    ptr = strtok_r(buf + sizeof(NUMPY_HEADER_STR) - 1, ":", &outer_end);

    // Read in the data type, order, and shape
    while (ptr && (!foundFormat || !foundOrder || !foundShape))
    {

        // End of string?
        if (!ptr)
            break;

        // Skip whitespace
        while (isspace(*ptr))
            ptr++;

        // Parse the dictionary field name
        if (!strcmp(ptr, "'descr'"))
        {
            ptr = strtok_r(NULL, ",", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            if (strcmp(ptr, dtype_str))
            {
                rc = FILE_TYPE_MISMATCH;
                goto done;
            }

            foundFormat = true;
        }
        else if (!strcmp(ptr, "'fortran_order'"))
        {
            ptr = strtok_r(NULL, ",", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            if (!strcmp(ptr, "False"))
            {
                fortranOrder = false;
            }
            else
            {
                rc = FILE_TYPE_MISMATCH;
                goto done;
            }

            foundOrder = true;
        }
        else if (!strcmp(ptr, "'shape'"))
        {

            ptr = strtok_r(NULL, "(", &outer_end);
            if (!ptr)
                break;
            ptr = strtok_r(NULL, ")", &outer_end);
            if (!ptr)
                break;

            while (isspace(*ptr))
                ptr++;

            // The shape contains N comma-separated integers. Read up to 4.
            char* end = NULL;

            ptr = strtok_r(ptr, ",", &end);
            for (int i = 0; i < 4; i++)
            {
                // Out of dimensions
                if (!ptr)
                    break;

                shape.push_back(atoi(ptr));
                totalElems *= atoi(ptr);
                ptr = strtok_r(NULL, ",", &end);
            }

            foundShape = true;
        }
        else
        {
            rc = HEADER_PARSE_ERROR;
            goto done;
        }

        if (!ptr)
            break;

        ptr = strtok_r(NULL, ":", &outer_end);
    }

    if (!foundShape || !foundFormat || !foundOrder)
    {
        rc = HEADER_PARSE_ERROR;
        goto done;
    }

    // Validate header
    if (fortranOrder != false)
    {
        rc = FILE_TYPE_MISMATCH;
        goto done;
    }

    if (totalElems != elems)
    {
        rc = BUFFER_SIZE_MISMATCH;
        goto done;
    }

    // Go back to the begininng and read until the end of the header dictionary
    rewind(infile);
    int val;

    do
    {
        val = fgetc(infile);
    } while (val != EOF && val != '\n');

done:

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const bool* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const bool* databuf)
{
    const char dtype_str[] = "'|b1'";
    FILE* outfile          = nullptr;
    NPError rc             = NO_ERROR;
    uint32_t totalElems    = 1;

    assert(filename);
    assert(shape.size() >= 0);
    assert(databuf);

    outfile = fopen(filename, "wb");

    if (!outfile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    for (uint32_t i = 0; i < shape.size(); i++)
    {
        totalElems *= shape[i];
    }

    rc = writeNpyHeader(outfile, shape, dtype_str);

    // Numpy save format stores booleans as a byte array
    // with one byte per boolean.  This somewhat inefficiently
    // remaps from system bool[] to this format.
    for (uint32_t i = 0; i < totalElems; i++)
    {
        int val = databuf[i] ? 1 : 0;
        if (fputc(val, outfile) == EOF)
        {
            rc = FILE_IO_ERROR;
            goto done;
        }
    }

done:

    if (outfile)
        fclose(outfile);

    return rc;
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const int32_t* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int32_t* databuf)
{
    const char dtype_str[] = "'<i4'";
    FILE* outfile          = nullptr;
    NPError rc             = NO_ERROR;
    uint32_t totalElems    = 1;

    assert(filename);
    assert(shape.size() >= 0);
    assert(databuf);

    outfile = fopen(filename, "wb");

    if (!outfile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    for (uint32_t i = 0; i < shape.size(); i++)
    {
        totalElems *= shape[i];
    }

    rc = writeNpyHeader(outfile, shape, dtype_str);

    if (fwrite(databuf, sizeof(int32_t), totalElems, outfile) != totalElems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (outfile)
        fclose(outfile);

    return rc;
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const int64_t* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int64_t* databuf)
{
    const char dtype_str[] = "'<i8'";
    FILE* outfile          = nullptr;
    NPError rc             = NO_ERROR;
    uint32_t totalElems    = 1;

    assert(filename);
    assert(shape.size() >= 0);
    assert(databuf);

    outfile = fopen(filename, "wb");

    if (!outfile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    for (uint32_t i = 0; i < shape.size(); i++)
    {
        totalElems *= shape[i];
    }

    rc = writeNpyHeader(outfile, shape, dtype_str);

    if (fwrite(databuf, sizeof(int64_t), totalElems, outfile) != totalElems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (outfile)
        fclose(outfile);

    return rc;
}

NumpyUtilities::NPError NumpyUtilities::writeToNpyFile(const char* filename, const uint32_t elems, const float* databuf)
{
    std::vector<int32_t> shape = { (int32_t)elems };
    return writeToNpyFile(filename, shape, databuf);
}

NumpyUtilities::NPError
    NumpyUtilities::writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const float* databuf)
{
    const char dtype_str[] = "'<f4'";
    FILE* outfile          = nullptr;
    NPError rc             = NO_ERROR;
    uint32_t totalElems    = 1;

    assert(filename);
    assert(shape.size() >= 0);
    assert(databuf);

    outfile = fopen(filename, "wb");

    if (!outfile)
    {
        rc = FILE_NOT_FOUND;
        goto done;
    }

    for (uint32_t i = 0; i < shape.size(); i++)
    {
        totalElems *= shape[i];
    }

    rc = writeNpyHeader(outfile, shape, dtype_str);

    if (fwrite(databuf, sizeof(float), totalElems, outfile) != totalElems)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    if (outfile)
        fclose(outfile);

    return rc;
}

NumpyUtilities::NPError
    NumpyUtilities::writeNpyHeader(FILE* outfile, const std::vector<int32_t>& shape, const char* dtype_str)
{
    NPError rc = NO_ERROR;
    uint32_t i;
    char header[NUMPY_HEADER_SZ + 1];
    int headerPos = 0;

    assert(outfile);
    assert(shape.size() >= 0);

    // Space-fill the header and end with a newline to start per numpy spec
    memset(header, 0x20, NUMPY_HEADER_SZ);
    header[NUMPY_HEADER_SZ - 1] = '\n';
    header[NUMPY_HEADER_SZ]     = 0;

    // Write out the hard-coded header.  We only support a 128-byte 1.0 header
    // for now, which should be sufficient for simple tensor types of any
    // reasonable rank.
    memcpy(header, NUMPY_HEADER_STR, sizeof(NUMPY_HEADER_STR) - 1);
    headerPos += sizeof(NUMPY_HEADER_STR) - 1;

    // Output the format dictionary
    // Hard-coded for I32 for now
    headerPos +=
        snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, "'descr': %s, 'fortran_order': False, 'shape': (%d,",
                 dtype_str, shape.size() > 0 ? shape[0] : 1);

    // Remainder of shape array
    for (i = 1; i < shape.size(); i++)
    {
        headerPos += snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, " %d,", shape[i]);
    }

    // Close off the dictionary
    headerPos += snprintf(header + headerPos, NUMPY_HEADER_SZ - headerPos, "), }");

    // snprintf leaves a NULL at the end. Replace with a space
    header[headerPos] = 0x20;

    if (fwrite(header, NUMPY_HEADER_SZ, 1, outfile) != 1)
    {
        rc = FILE_IO_ERROR;
        goto done;
    }

done:

    return rc;
}
