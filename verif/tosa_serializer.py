

# Copyright (c) 2020-2021, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#!/usr/bin/env python3

import flatbuffers
import numpy as np
from enum import Enum, IntEnum, unique
from tosa import TosaGraph, TosaBasicBlock, TosaTensor, TosaOperator, DType, Format, Usage, Op, ResizeMode, Version
import tosa
import os
import json

# With the way flatc generates its python types, there is no programatic way
# to get string names for the integer types.  Manually maintain a string table
# here.
DTypeNames = [ 'UNKNOWN',
               'BOOL',
               'UINT8',
               'INT4',
               'INT8',
               'INT16',
               'INT32',
               'INT48',
               'FLOAT' ]

def dtype_str_to_val(name):

    for i in range(len(DTypeNames)):
        if name.casefold() == DTypeNames[i].casefold():
            return i
    raise Exception('Unable to parse DType name {}'.format(name))


class TosaSerializerUnion:
    '''This class handles encapsulating and serializing union types into flatbuffers'''
    def __init__(self):

        # A tuple of the start and end functions.  Set by the options constructors below
        self.optFcns = None

        # The type from the tosa.Options enumeration.  Set by the options constructors below.
        self.utype = None

        # Each of these lists is a tuple of the add function and the
        # value being added.  Set by the options constructors below.
        self.ints = []
        self.bools = []
        self.floats = []
        self.strings = []
        self.intvecs = []
        self.fpvecs = []

    def serialize(self, builder):

        # We have to build strings and vectors first
        strList = []
        intVecList = []
        fpVecList = []

        for fcn, val in self.strings:
            strList.append((fcn, builder.CreateString(val)))

        for fcn, val in self.intvecs:
            intVecList.append((fcn, TosaSerializer.serializeInt32Vec(builder, val)))

        for fcn, val in self.fpvecs:
            fpVecList.append((fcn, TosaSerializer.serializeFpVec(builder, val)))

        startFcn, endFcn = self.optFcns

        # Then serialize the options object from the list of primitives and
        # other serialized values
        startFcn(builder)
        for fcn, val in self.ints:
            fcn(builder, val)

        for fcn, val in self.bools:
            fcn(builder, val)

        for fcn, val in self.floats:
            fcn(builder, val)

        for fcn, val in strList:
            fcn(builder, val)

        for fcn, val in intVecList:
            fcn(builder, val)

        for fcn, val in fpVecList:
            fcn(builder, val)

        return endFcn(builder)

class TosaSerializerAttribute(TosaSerializerUnion):
    '''This class handles encapsulating all of the enumerated types for attributes'''

    def __init__(self):
        super().__init__()

    def Pool2dAttribute(self, kernel, stride, padding):
        from tosa import Pool2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().Pool2dAttribute

        self.optFcns = (a.Pool2dAttributeStart, a.Pool2dAttributeEnd)
        self.intvecs.append((a.Pool2dAttributeAddPadding,
                             padding))
        self.intvecs.append((a.Pool2dAttributeAddKernel,
                             kernel))
        self.intvecs.append((a.Pool2dAttributeAddStride,
                             stride))

    def Conv2dAttribute(self, padding, stride, dilation):
        from tosa import Conv2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().Conv2dAttribute
        self.optFcns = (a.Conv2dAttributeStart, a.Conv2dAttributeEnd)

        self.intvecs.append((a.Conv2dAttributeAddPadding,
                             padding))
        self.intvecs.append((a.Conv2dAttributeAddStride,
                             stride))
        self.intvecs.append((a.Conv2dAttributeAddDilation,
                             dilation))

    def TransposeConv2DAttribute(self, outpad, stride, dilation, output_shape):
        from tosa import TransposeConv2dAttribute as a, Attribute

        self.utype = Attribute.Attribute().TransposeConv2dAttribute
        self.optFcns = (a.TransposeConv2dAttributeStart, a.TransposeConv2dAttributeEnd)

        self.intvecs.append((a.TransposeConv2dAttributeAddOutpad,
                             outpad))
        self.intvecs.append((a.TransposeConv2dAttributeAddStride,
                             stride))
        self.intvecs.append((a.TransposeConv2dAttributeAddDilation,
                             dilation))
        self.intvecs.append((a.TransposeConv2dAttributeAddOutputShape,
                             output_shape))

    def ReluNAttribute(self, maxint, maxfp):
        from tosa import ReluNAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReluNAttribute
        self.optFcns = (a.ReluNAttributeStart, a.ReluNAttributeEnd)

        self.ints.append((a.ReluNAttributeAddMaxInt, maxint))
        self.ints.append((a.ReluNAttributeAddMaxFp, maxfp))


    def AxisAttribute(self, axis):
        from tosa import AxisAttribute as a, Attribute

        self.utype = Attribute.Attribute().AxisAttribute
        self.optFcns = (a.AxisAttributeStart, a.AxisAttributeEnd)

        self.ints.append((a.AxisAttributeAddAxis,
                          axis))

    def ReshapeAttribute(self, shape):
        from tosa import ReshapeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ReshapeAttribute
        self.optFcns = (a.ReshapeAttributeStart, a.ReshapeAttributeEnd)

        self.intvecs.append((a.ReshapeAttributeAddShape,
                             shape))

    def SliceAttribute(self, begin, size):
        from tosa import SliceAttribute as a, Attribute

        self.utype = Attribute.Attribute().SliceAttribute
        self.optFcns = (a.SliceAttributeStart, a.SliceAttributeEnd)

        self.intvecs.append((a.SliceAttributeAddBegin,
                             begin))
        self.intvecs.append((a.SliceAttributeAddSize,
                             size))

    def TileAttribute(self, multiples):
        from tosa import TileAttribute as a, Attribute

        self.utype = Attribute.Attribute().TileAttribute
        self.optFcns = (a.TileAttributeStart, a.TileAttributeEnd)

        self.intvecs.append((a.TileAttributeAddMultiples,
                             multiples))

    def ResizeAttribute(self, output_size, stride, offset, shift, stride_fp, offset_fp, mode):
        from tosa import ResizeAttribute as a, Attribute

        self.utype = Attribute.Attribute().ResizeAttribute
        self.optFcns = (a.ResizeAttributeStart, a.ResizeAttributeEnd)

        self.intvecs.append((a.ResizeAttributeAddOutputSize,
                             output_size))
        self.intvecs.append((a.ResizeAttributeAddStride,
                             stride))
        self.intvecs.append((a.ResizeAttributeAddOffset,
                             offset))
        self.ints.append((a.ResizeAttributeAddShift,
                         shift))
        self.fpvecs.append((a.ResizeAttributeAddStrideFp,
                            stride_fp))
        self.fpvecs.append((a.ResizeAttributeAddOffsetFp,
                            offset_fp))
        self.ints.append((a.ResizeAttributeAddMode,
                         mode))

    def ClampAttribute(self, minint, maxint, minfp, maxfp):
        from tosa import ClampAttribute as a, Attribute

        self.utype = Attribute.Attribute().ClampAttribute
        self.optFcns = (a.ClampAttributeStart, a.ClampAttributeEnd)

        self.ints.append((a.ClampAttributeAddMinInt,
                          minint))
        self.ints.append((a.ClampAttributeAddMaxInt,
                          maxint))

        self.ints.append((a.ClampAttributeAddMinFp,
                          minfp))
        self.ints.append((a.ClampAttributeAddMaxFp,
                          maxfp))

    def RescaleAttribute(self, input_zp, output_zp, multiplier, shift, scale32, double_round, per_channel):
        from tosa import RescaleAttribute as a, Attribute

        self.utype = Attribute.Attribute().RescaleAttribute
        self.optFcns = (a.RescaleAttributeStart, a.RescaleAttributeEnd)

        self.ints.append((a.RescaleAttributeAddInputZp,
                         input_zp))
        self.ints.append((a.RescaleAttributeAddOutputZp,
                         output_zp))
        self.intvecs.append((a.RescaleAttributeAddMultiplier,
                             multiplier))
        self.intvecs.append((a.RescaleAttributeAddShift,
                             shift))
        self.bools.append((a.RescaleAttributeAddScale32,
                           scale32))
        self.bools.append((a.RescaleAttributeAddDoubleRound,
                           double_round))
        self.bools.append((a.RescaleAttributeAddPerChannel,
                           per_channel))

    def MulAttribute(self, shift):
        from tosa import MulAttribute as a, Attribute

        self.utype = Attribute.Attribute().MulAttribute
        self.optFcns = (a.MulAttributeStart, a.MulAttributeEnd)

        self.ints.append((a.MulAttributeAddShift,
                         shift))

    def ArithmeticRightShiftAttribute(self, round):
        from tosa import ArithmeticRightShiftAttribute as a, Attribute

        self.utype = Attribute.Attribute().ArithmeticRightShiftAttribute
        self.optFcns = (a.ArithmeticRightShiftAttributeStart, a.ArithmeticRightShiftAttributeEnd)

        self.bools.append((a.ArithmeticRightShiftAttributeAddRound,
                         round))

    def CustomAttribute(self, identifier):
        from tosa import CustomAttribute as a, Attribute

        self.utype = Attribute.Attribute().CustomAttribute
        self.optFcns = (a.CustomAttributeStart, a.CustomAttributeEnd)

        self.strings.append((a.CustomAttributeAddIdentifier,
                         identifier))

    def CondIfAttribute(self, then_branch, else_branch):
        from tosa import CondIfAttribute as a, Attribute

        self.utype = Attribute.Attribute().CondIfAttribute
        self.optFcns = (a.CondIfAttributeStart, a.CondIfAttributeEnd)

        self.strings.append((a.CondIfAttributeAddThenBranch,
                         then_branch))
        self.strings.append((a.CondIfAttributeAddElseBranch,
                         else_branch))

    def WhileLoopAttribute(self, cond_branch, body_branch):
        from tosa import WhileLoopAttribute as a, Attribute

        self.utype = Attribute.Attribute().WhileLoopAttribute
        self.optFcns = (a.WhileLoopAttributeStart, a.WhileLoopAttributeEnd)

        self.strings.append((a.WhileLoopAttributeAddCondBranch,
                         cond_branch))
        self.strings.append((a.WhileLoopAttributeAddBodyBranch,
                         body_branch))

class TosaSerializerQuantInfo(TosaSerializerUnion):
    '''This class handles encapsulating all of the enumerated types for quantinfo types'''
    def __init__(self):
        super().__init__()

    def ConvQuantInfo(self, input_zp, weight_zp):
        from tosa import ConvQuantInfo as q, QuantInfo

        self.utype = QuantInfo.QuantInfo().ConvQuantInfo
        self.optFcns = (q.ConvQuantInfoStart, q.ConvQuantInfoEnd)
        self.ints.append((q.ConvQuantInfoAddInputZp, input_zp))
        self.ints.append((q.ConvQuantInfoAddWeightZp, weight_zp))

    def UnaryQuantInfo(self, input_zp, output_zp):
        from tosa import UnaryQuantInfo as q, QuantInfo

        self.utype = QuantInfo.QuantInfo().UnaryQuantInfo
        self.optFcns = (q.UnaryQuantInfoStart, q.UnaryQuantInfoEnd)
        self.ints.append((q.UnaryQuantInfoAddInputZp, input_zp))
        self.ints.append((q.UnaryQuantInfoAddOutputZp, output_zp))

    def MatMulQuantInfo(self, a_zp, b_zp):
        from tosa import MatMulQuantInfo as q, QuantInfo

        self.utype = QuantInfo.QuantInfo().MatMulQuantInfo
        self.optFcns = (q.MatMulQuantInfoStart, q.MatMulQuantInfoEnd)
        self.ints.append((q.MatMulQuantInfoAddAZp, a_zp))
        self.ints.append((q.MatMulQuantInfoAddBZp, b_zp))

    def PadQuantInfo(self, input_zp):
        from tosa import PadQuantInfo as q, QuantInfo

        self.utype = QuantInfo.QuantInfo().PadQuantInfo
        self.optFcns = (q.PadQuantInfoStart, q.PadQuantInfoEnd)
        self.ints.append((q.PadQuantInfoAddInputZp, input_zp))

class TosaSerializerTensor:
    def __init__(self, name, shape, dtype, usage, dformat, filename = None, placeholderFilename = None):
        self.name = name

        if isinstance(shape, np.ndarray):
            shape = shape.astype(int).tolist()
        shape = list(map(int, shape))

        self.shape = shape
        self.dtype = dtype
        self.usage = TosaSerializer.toList(usage)
        self.dformat = TosaSerializer.toList(dformat)

        # Filename for const tensors.  This gets written to the .tosa serialization
        self.filename = filename

        # Filename for placeholder tensors.  These get generated by the test generation
        # process and are written to disk, but are considered input tensors by the network
        # so they do not appear in the TOSA serialiazation.  However, if we want to form a unit
        # test around these input tensors, we can get the filename from here.
        self.placeholderFilename = placeholderFilename

    def __str__(self):
        str = 'TosaSerializerTensor name: {} shape: {} dtype: {} Usage: {} format {} filename: {}'.format(
            self.name, self.shape, DTypeNames[self.dtype], self.usage, self.dformat, self.filename)
        return str

    def addUsage(self, usage):
        self.usage.append(usage)

    def addFormat(self, format):
        self.dformat.append(format)

    def setDtype(self, dtype):
        self.dtype = dtype

    def merge(self, name, shape, dtype, usage, dformat, filename = None):
        # Merge in additional usage/formats to the list
        found = 0
        for i in self.usage:
            if i == usage:
                found = 1
                break
        if not found:
            self.usage.append(usage)

        found = 0
        for i in self.dformat:
            if i == dformat:
                found = 1
                break
        if not found:
            self.dformat.append(dformat)

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        if self.filename:
            fb_filename = builder.CreateString(self.filename)
        fb_shapes = TosaSerializer.serializeInt32Vec(builder, self.shape)
        fb_usage = TosaSerializer.serializeInt32Vec(builder, self.usage)
        fb_dformat = TosaSerializer.serializeInt32Vec(builder, self.dformat)

        TosaTensor.TosaTensorStart(builder)
        TosaTensor.TosaTensorAddName(builder, fb_name)
        TosaTensor.TosaTensorAddShape(builder, fb_shapes)
        TosaTensor.TosaTensorAddType(builder, self.dtype)
        TosaTensor.TosaTensorAddUsage(builder, fb_usage)
        TosaTensor.TosaTensorAddFormat(builder, fb_dformat)
        if self.filename:
            TosaTensor.TosaTensorAddNpyFilename(builder, fb_filename)

        return TosaTensor.TosaTensorEnd(builder)

class TosaSerializerOperator:
    def __init__(self, op, inputs, outputs, attributes = None, quantInfo = None):
        self.op = op
        self.attributes = attributes
        self.inputs = TosaSerializer.toList(inputs)
        self.outputs = TosaSerializer.toList(outputs)
        self.quantInfo = quantInfo

    def __str__(self):
        str = 'Op {}\n----\n'.format(self.op)

        for i in self.inputs:
            str = str + '  Input:  {}\n'.format(i)
        for o in self.outputs:
            str = str + '  Output: {}\n'.format(o)

        return str

    def serialize(self, builder):
        fb_inputs = TosaSerializer.serializeStrVec(builder, self.inputs, TosaOperator.TosaOperatorStartInputsVector)
        fb_outputs = TosaSerializer.serializeStrVec(builder, self.outputs, TosaOperator.TosaOperatorStartOutputsVector)
        # Need to serialize quant_info and attributes enums still
        if self.attributes is not None:
            fb_attributes = self.attributes.serialize(builder)

        if self.quantInfo is not None:
            fb_qinfo = self.quantInfo.serialize(builder)

        TosaOperator.TosaOperatorStart(builder)
        TosaOperator.TosaOperatorAddOp(builder, self.op)
        TosaOperator.TosaOperatorAddInputs(builder, fb_inputs)
        TosaOperator.TosaOperatorAddOutputs(builder, fb_outputs)
        if self.attributes is not None:
            TosaOperator.TosaOperatorAddAttributeType(builder, self.attributes.utype)
            TosaOperator.TosaOperatorAddAttribute(builder, fb_attributes)
        if self.quantInfo is not None:
            TosaOperator.TosaOperatorAddQuantInfoType(builder, self.quantInfo.utype)
            TosaOperator.TosaOperatorAddQuantInfo(builder, fb_qinfo)

        return TosaOperator.TosaOperatorEnd(builder)

class TosaSerializerBasicBlock:
    def __init__(self, name):
        self.name = name
        self.operators = []

        # Dict assures uniqueness, but allows us to look up by name
        self.tensors = dict()

        self.inputs = []
        self.outputs = []

    def addTensor(self, name, shape, dtype, usage, dformat, filename = None, placeholderFilename = None):
        try:
            # Someone already added this tensor.
            # We may have to add more usages and formats
            tens = self.tensors[name]
            filename = tens.merge(name, shape, dtype, usage, dformat, filename)
        except KeyError:
            self.tensors[name] = TosaSerializerTensor(name, shape, dtype, usage, dformat, filename, placeholderFilename)

        return self.tensors[name]

    def addInput(self, name):
        self.inputs.append(name)

    def addOutput(self, name):
        self.outputs.append(name)

    def addOperator(self, op, inputs, outputs, attributes = None, quant_info = None):
        self.operators.append(TosaSerializerOperator(op, inputs, outputs, attributes, quant_info))

    def serialize(self, builder):
        fb_name = builder.CreateString(self.name)
        fbv_inputs  = TosaSerializer.serializeStrVec(builder, list(self.inputs), TosaBasicBlock.TosaBasicBlockStartInputsVector)
        fbv_outputs = TosaSerializer.serializeStrVec(builder, list(self.outputs), TosaBasicBlock.TosaBasicBlockStartOutputsVector)
        fbv_tensors = TosaSerializer.serializeObjVec(builder, list(self.tensors.values()), TosaBasicBlock.TosaBasicBlockStartTensorsVector)
        fbv_operators = TosaSerializer.serializeObjVec(builder, self.operators, TosaBasicBlock.TosaBasicBlockStartOperatorsVector)

        TosaBasicBlock.TosaBasicBlockStart(builder)
        TosaBasicBlock.TosaBasicBlockAddName(builder, fb_name)
        TosaBasicBlock.TosaBasicBlockAddInputs(builder, fbv_inputs)
        TosaBasicBlock.TosaBasicBlockAddOutputs(builder, fbv_outputs)
        TosaBasicBlock.TosaBasicBlockAddTensors(builder, fbv_tensors)
        TosaBasicBlock.TosaBasicBlockAddOperators(builder, fbv_operators)
        return TosaBasicBlock.TosaBasicBlockEnd(builder)

@unique
class TensorDir(IntEnum):
    PLACEHOLDER = 0
    CONST = 1
    INTERMEDIATE = 2
    RESULT = 3

class TosaSerializer:
    def __init__(self, pathPrefix):

        # Get the global TOSA version if not already defined
        try:
            TOSA_VERSION
        except NameError:
            TosaSerializer.setTosaVersion()

        self.builder = flatbuffers.Builder(0)

        self.basicBlocks = []
        self.startBasicBlock('main')
        self.pathPrefix = pathPrefix

        # Indicies used for adding/naming tensors
        self.currInputIdx = 0
        self.currConstIdx = 0
        self.currLayerIdx = 1
        self.currResultIdx = 0

        # Is this an illegal test that is expected to fail?
        self.expectedFailure = False
        self.expectedFailureDesc = ''

    def __str__(self):
        str = ''
        for bb in self.basicBlocks:
            str = str + bb.__str__()
        return str

    def addPlaceholder(self, shape, dtype, usage, dformat, vals):
        if not self.currBasicBlock:
            raise Exception('addTensor called without valid basic block')

        name = 'input-{}'.format(self.currInputIdx)
        filename = '{}.npy'.format(name)
        self.currInputIdx = self.currInputIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, usage, dformat, None, filename)
        # This is always an input to the block
        self.currBasicBlock.addInput(name)
        # Add the operator now
        self.currBasicBlock.addOperator(tosa.Op.Op().PLACEHOLDER, [], name)

        if vals is not None:
            np.save(os.path.join(self.pathPrefix, filename), vals, False)

        return tens

    def addConst(self, shape, dtype, usage, dformat, vals):
        if not self.currBasicBlock:
            raise Exception('addTensor called without valid basic block')

        name = 'const-{}'.format(self.currInputIdx)
        filename = '{}.npy'.format(name)
        self.currInputIdx = self.currInputIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, usage, dformat, filename)
        # Add the operator now
        self.currBasicBlock.addOperator(tosa.Op.Op().CONST, [], name)

        if vals is not None:
            np.save(os.path.join(self.pathPrefix, filename), vals, False)
        return tens

    def addIntermediate(self, shape, dtype, usage, dformat):

        if not self.currBasicBlock:
            raise Exception('addTensor called without valid basic block')

        name = 'layer-{}'.format(self.currLayerIdx)
        filename = None # No file, so no filename
        self.currLayerIdx = self.currLayerIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, usage, dformat, filename)

        return tens

    def addInputTensor(self, tensor):
        self.currBasicBlock.addOperator(tosa.Op.Op().PLACEHOLDER, [], tensor.name)
        self.currBasicBlock.addTensor(tensor.name, tensor.shape, tensor.dtype, tensor.usage, tensor.dformat)
        self.currBasicBlock.addInput(tensor.name)

    def addOutputTensor(self, tensor):
        self.currBasicBlock.addOutput(tensor.name)

    def addOutput(self, shape, dtype, usage, dformat):
        if not self.currBasicBlock:
            raise Exception('addTensor called without valid basic block')

        name = 'result-{}'.format(self.currResultIdx)
        self.currResultIdx = self.currResultIdx + 1

        tens = self.currBasicBlock.addTensor(name, shape, dtype, usage, dformat, None)
        self.currBasicBlock.addOutput(name)
        return tens

    def addOperator(self, op, inputs, outputs, attributes = None, quant_info = None):

        if op == tosa.Op.Op().PLACEHOLDER or \
           op == tosa.Op.Op().CONST:
            raise Exception('Use addPlaceholderTensor() or addConstTensor() to add PLACEHOLDER and CONST ops')

        return self.currBasicBlock.addOperator(op, inputs, outputs, attributes, quant_info)

    def setExpectedFailure(self, desc='', val=True):
        self.expectedFailure = val
        self.expectedFailureDesc = desc

    def setExpectedFailure(self, desc='', val=True):
        self.expectedFailure = val
        self.expectedFailureDesc = desc

    def serialize(self):

        builder = self.builder

        Version.VersionStart(builder)
        Version.VersionAdd_major(builder, TOSA_VERSION[0])
        Version.VersionAdd_minor(builder, TOSA_VERSION[1])
        Version.VersionAdd_patch(builder, TOSA_VERSION[2])
        Version.VersionAdd_experimental(builder, TOSA_VERSION[3])
        version = Version.VersionEnd(builder)

        fbv_bb = TosaSerializer.serializeObjVec(builder, self.basicBlocks, TosaGraph.TosaGraphStartBlocksVector)

        TosaGraph.TosaGraphStart(builder)
        TosaGraph.TosaGraphAddVersion(builder, version)
        TosaGraph.TosaGraphAddBlocks(builder, fbv_bb)
        graph = TosaGraph.TosaGraphEnd(builder)

        self.builder.Finish(graph)
        return self.builder.Output()

    def writeJson(self, tosa_filename):
        '''Write a json test file so that it is fairly easy to pick up the test
           and generate commands for third party tool'''
        test_desc = dict()

        test_desc['tosa_file'] = tosa_filename
        ifm_name = []
        ifm_shape = []
        ifm_file = []
        ofm_name = []
        ofm_file = []
        ofm_shape = []

        for b in self.basicBlocks:
            if b.name == 'main':
                for i in b.inputs:
                    ifm_name.append(i)
                    ifm_shape.append(b.tensors[i].shape)
                    ifm_file.append(b.tensors[i].placeholderFilename)
                for o in b.outputs:
                    ofm_name.append(o)
                    ofm_shape.append(b.tensors[o].shape)
                    # Make up an OFM filename here.  One isn't generated until the reference tool is
                    # run, so any name is a good name
                    ofm_file.append('ref-{}.npy'.format(o))

        test_desc['ifm_placeholder'] = ifm_name
        test_desc['ifm_file'] = ifm_file
        test_desc['ifm_shape'] = ifm_shape
        test_desc['ofm_name'] = ofm_name
        test_desc['ofm_shape'] = ofm_shape
        test_desc['ofm_file'] = ofm_file
        test_desc['expected_failure'] = self.expectedFailure
        if self.expectedFailureDesc:
            test_desc['expected_failure_desc'] = self.expectedFailureDesc

        return json.dumps(test_desc, indent='  ')

    def startBasicBlock(self, name):
        self.currBasicBlock = TosaSerializerBasicBlock(name)
        self.basicBlocks.append(self.currBasicBlock)

    @staticmethod
    def serializeStrVec(builder, vec, start_fcn):
        fb_strs = [builder.CreateString(i) for i in vec]
        start_fcn(builder, len(fb_strs))
        for s in fb_strs[::-1]:
            builder.PrependUOffsetTRelative(s)
        return builder.EndVector(len(fb_strs))

    @staticmethod
    def serializeInt32Vec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependInt32(v)
        return builder.EndVector(len(vec))

    @staticmethod
    def serializeFpVec(builder, vec):
        builder.StartVector(4, len(vec), 4)
        for v in vec[::-1]:
            builder.PrependFloat32(v)
        return builder.EndVector(len(vec))

    @staticmethod
    def serializeObjVec(builder, vec, start_fcn):
        serialized_vec = []
        for v in vec[::-1]:
            serialized_vec.append(v.serialize(builder))

        start_fcn(builder, len(vec))
        for v in serialized_vec:
            builder.PrependUOffsetTRelative(v)
        return builder.EndVector(len(vec))

    @staticmethod
    def toList(val):
        if isinstance(val, list):
            return val
        else:
            return [val]

    @staticmethod
    def setTosaVersion():
        # Create a dummy flatbuffers file with the default version information
        # There does not appear to be a better way to get a constant from a
        # flatbuffer schema file
        builder = flatbuffers.Builder(0)
        Version.VersionStart(builder)
        ver = Version.VersionEnd(builder)
        TosaGraph.TosaGraphStart(builder)
        TosaGraph.TosaGraphAddVersion(builder, ver)
        gr = TosaGraph.TosaGraphEnd(builder)
        builder.Finish(gr)

        out = builder.Output()

        gr = TosaGraph.TosaGraph()
        root = gr.GetRootAsTosaGraph(out, 0)

        # Store the version as a global variable so that it only needs to be
        # generated once per process.
        global TOSA_VERSION
        TOSA_VERSION = [root.Version()._major(),
                        root.Version()._minor(),
                        root.Version()._patch(),
                        root.Version()._experimental() ]
