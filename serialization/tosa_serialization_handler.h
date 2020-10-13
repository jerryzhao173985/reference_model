
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

#ifndef _TOSA_SERIALIZATION_HANDLER_H
#define _TOSA_SERIALIZATION_HANDLER_H
#include "attribute.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "quant_info.h"
#include "tosa_generated.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tosa
{

enum tosa_err_t
{
    TOSA_OK,
    TOSA_USER_ERROR,
    TOSA_FILE_ERROR,
    TOSA_MEMORY_ERROR,
    TOSA_SCHEMA_MISSING,
    TOSA_INTERNAL_ERROR,
    TOSA_VERSION_MISMATCH,
    NUM_TOSA_ERROR
};

struct TosaVersion
{
    int32_t _major;
    int32_t _minor;
    int32_t _patch;
    bool _experimental;

    TosaVersion() = delete;
    TosaVersion(int32_t major, int32_t minor, int32_t patch, bool experimental)
    {
        _major        = major;
        _minor        = minor;
        _patch        = patch;
        _experimental = experimental;
    }

    std::string to_string() const
    {
        std::string str;
        str += std::to_string(_major) + ".";
        str += std::to_string(_minor) + ".";
        str += std::to_string(_patch);
        if (_experimental)
            str += "(experimental)";
        return str;
    };

    bool operator==(const TosaVersion& rhs)
    {
        if (rhs._major == _major && rhs._minor == _minor && rhs._patch == _patch && rhs._experimental == _experimental)
        {
            return true;
        }
        return false;
    }

    bool operator!=(const TosaVersion& rhs)
    {
        return !((*this) == rhs);
    }
};

class TosaSerializationHandler;

class TosaSerializationTensor
{
public:
    // constructor and destructor
    TosaSerializationTensor(const flatbuffers::String* name,
                            const flatbuffers::Vector<uint32_t>& usage,
                            const flatbuffers::Vector<int32_t>& shape,
                            DType dtype,
                            const flatbuffers::Vector<uint32_t>& format,
                            const flatbuffers::String* npy_filename);
    TosaSerializationTensor(std::string name,
                            const std::vector<Usage>& usage,
                            const std::vector<int32_t>& shape,
                            DType dtype,
                            const std::vector<Format>& format,
                            const std::string* npy_filename);
    TosaSerializationTensor();
    ~TosaSerializationTensor();

    // copy constructor/assignment
    TosaSerializationTensor(const TosaSerializationTensor& rhs);
    TosaSerializationTensor& operator=(const TosaSerializationTensor& rhs);

    // move constructor/assignment
    TosaSerializationTensor(TosaSerializationTensor&& rhs);
    TosaSerializationTensor& operator=(TosaSerializationTensor&& rhs);

    // accessor
    std::string GetName() const
    {
        return *_name;
    }
    const std::vector<int32_t>& GetShape() const
    {
        return *_shape;
    }
    DType GetDtype()
    {
        return _dtype;
    }
    bool HasFormat(Format format)
    {
        for (Format us : *_format)
        {
            if (us == format)
                return true;
        }
        return false;
    }
    std::vector<Format>& GetFormat()
    {
        return *_format;
    }
    bool HasUsage(Usage usage)
    {
        for (Usage us : *_usage)
        {
            if (us == usage)
                return true;
        }
        return false;
    }
    std::vector<Usage>& GetUsage()
    {
        return *_usage;
    }
    std::string* GetNpyFilePtr() const
    {
        return _npy_filename;
    }

    // modifier
    void SetDtype(DType dtype)
    {
        _dtype = dtype;
    }
    void SetName(std::string name)
    {
        *_name = name;
    }

private:
    DType _dtype;                 /* data type enumeration, see tosa_isa_generated.h */
    std::vector<Format>* _format; /* list of possible tensor format */
    std::vector<Usage>* _usage;   /* list of possible tensor usage */
    std::vector<int32_t>* _shape; /* shape of the tensor */
    std::string* _name;           /* name of the tensor, used for solving dependency */
    std::string* _npy_filename;   /* numpy array filename if not null. so null is the distinguisher */
};

class TosaSerializationOperator
{
public:
    // use default copy, void constructor
    // constructor and destructor
    TosaSerializationOperator(Op op_name,
                              Attribute attribute_type,
                              const TosaAttributeBase* attribute,
                              QuantInfo qinfo_type,
                              const TosaQuantInfoBase* qinfo,
                              std::vector<std::string> input_tensor_names,
                              std::vector<std::string> output_tensor_names);
    ~TosaSerializationOperator();

    // accessor
    Op GetOp() const
    {
        return _op;
    }
    Attribute GetAttributeType() const
    {
        return _attribute_type;
    }
    TosaAttributeBase* GetAttribute() const
    {
        return _attribute;
    }
    QuantInfo GetQInfoType() const
    {
        return _qinfo_type;
    }
    TosaQuantInfoBase* GetQInfo() const
    {
        return _qinfo;
    }
    std::vector<std::string>& GetInputTensorNames() const
    {
        return *_input_tensor_names;
    }
    std::vector<std::string>& GetOutputTensorNames() const
    {
        return *_output_tensor_names;
    }
    std::vector<TosaSerializationTensor*>& GetInputTensors() const
    {
        return *_input_tensors;
    }
    std::vector<TosaSerializationTensor*>& GetOutputTensors() const
    {
        return *_output_tensors;
    }

private:
    Op _op;                        /* operator enum, see tosa_isa_generated.h for enumeration table */
    Attribute _attribute_type;     /* operator attribute enum, used for dynamic casting TosaAttributeBase class */
    TosaAttributeBase* _attribute; /* real attribute class goes here */
    QuantInfo _qinfo_type;         /* QuantInfo enum */
    TosaQuantInfoBase* _qinfo;     /* base class pointer of QuantInfo */
    std::vector<std::string>* _input_tensor_names;  /* array of input tensor names */
    std::vector<std::string>* _output_tensor_names; /* array of output tensor names */

    std::vector<TosaSerializationTensor*>* _input_tensors;  /* array of input TosaSerializationTensor */
    std::vector<TosaSerializationTensor*>* _output_tensors; /* array of output TosaSerializationTensor */
};

class TosaSerializationBasicBlock
{
public:
    // constructor and destructor
    TosaSerializationBasicBlock(std::string name,
                                std::vector<TosaSerializationOperator*> operators,
                                std::vector<TosaSerializationTensor*> tensors,
                                std::vector<std::string> inputs,
                                std::vector<std::string> outputs);
    ~TosaSerializationBasicBlock();

    // accessor
    std::string GetName() const
    {
        return *_name;
    }
    std::vector<TosaSerializationOperator*>& GetOperators()
    {
        return *_operators;
    }
    std::vector<TosaSerializationTensor*>& GetTensors()
    {
        return *_tensors;
    }

    TosaSerializationTensor* GetTensorByName(std::string name)
    {
        TosaSerializationTensor* result = nullptr;
        for (auto tensor : GetTensors())
        {
            if (tensor->GetName() == name)
            {
                result = tensor;
                break;
            }
        }
        return result;
    }

    std::vector<std::string>& GetInputs()
    {
        return *_inputs;
    }
    std::vector<std::string>& GetOutputs()
    {
        return *_outputs;
    }

private:
    std::string* _name;                                  /* name of basic block */
    std::vector<TosaSerializationOperator*>* _operators; /* TosaSerializationOperator list */
    std::vector<TosaSerializationTensor*>* _tensors;     /* TosaSerializationTensor list */
    std::vector<std::string>* _inputs;                   /* array of string to specify block inputs */
    std::vector<std::string>* _outputs;                  /* array of string to specify block outputs */
};

/*
 * this is a helper class for writing/reading Tosa ISA
 * supported format: .tosa (flatbuffer), .json
 * and provide high-level std::vector-like interface
 * to access internal data structure
 */
class TosaSerializationHandler
{
public:
    // constructor and destructor
    TosaSerializationHandler();
    ~TosaSerializationHandler();

    // file io
    tosa_err_t LoadFileJson(const char* filename);
    tosa_err_t LoadFileTosaFlatbuffer(const char* filename);
    tosa_err_t SaveFileJson(const char* filename);
    tosa_err_t SaveFileTosaFlatbuffer(const char* filename);
    tosa_err_t LoadFileSchema(const char* filename);

    // version
    TosaVersion* GetTosaVersion() const
    {
        return _version;
    }

    // accessor
    std::vector<TosaSerializationBasicBlock*>& GetBlocks()
    {
        return *_blocks;
    }

    TosaSerializationBasicBlock* GetBlockByName(std::string name)
    {
        TosaSerializationBasicBlock* result = nullptr;
        for (auto block : GetBlocks())
        {
            if (block->GetName() == name)
            {
                result = block;
                break;
            }
        }
        return result;
    }
    TosaSerializationBasicBlock* GetMainBlock()
    {
        TosaSerializationBasicBlock* main_block = GetBlockByName(std::string("main"));
        assert(main_block);
        return main_block;
    }

    std::vector<std::string>& GetInputs()
    {
        return GetMainBlock()->GetInputs();
    }
    std::vector<std::string>& GetOutputs()
    {
        return GetMainBlock()->GetOutputs();
    }

    bool GetSchemaLoaded() const
    {
        return _schemaLoaded;
    }

protected:
    tosa_err_t Clear();
    tosa_err_t InitWithBuf(const uint8_t* buf);
    tosa_err_t FreezeBuilder();
    tosa_err_t SetTosaVersion();
    tosa_err_t CheckTosaVersion(const TosaVersion& read_version);

private:
    TosaVersion* _version;                              /* tosa version */
    flatbuffers::FlatBufferBuilder* _builder;           /* flatbuffer builder */
    flatbuffers::Parser* _parser;                       /* flatbuffer parser, used for json parsing */
    std::vector<TosaSerializationBasicBlock*>* _blocks; /* array structure to store all TosaSerializationBasicBlock */
    bool _schemaLoaded;                                 /* is the schema properly loaded? */
};

class NumpyUtilities
{
public:
    enum NPError
    {
        NO_ERROR = 0,
        FILE_NOT_FOUND,
        FILE_IO_ERROR,
        FILE_TYPE_MISMATCH,
        HEADER_PARSE_ERROR,
        BUFFER_SIZE_MISMATCH,
    };

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, float* buf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, int32_t* buf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, int64_t* buf);

    static NPError readFromNpyFile(const char* filename, const uint32_t elems, bool* buf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const bool* buf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const bool* buf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int32_t* buf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const int32_t* buf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const int64_t* buf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const int64_t* buf);

    static NPError writeToNpyFile(const char* filename, const std::vector<int32_t>& shape, const float* buf);

    static NPError writeToNpyFile(const char* filename, const uint32_t elems, const float* buf);

private:
    static NPError checkNpyHeader(FILE* infile, const uint32_t elems, const char* dtype_str);
    static NPError writeNpyHeader(FILE* infile, const std::vector<int32_t>& shape, const char* dtype_str);
};

}    // namespace tosa

#endif    // _TOSA_SERIALIZATION_HANDLER_H
