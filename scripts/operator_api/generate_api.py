"""Generate extended reference model API with eager operator execution entrypoints"""
# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import subprocess
from xml.dom import minidom

from jinja2 import Environment
from jinja2 import FileSystemLoader

# Note: main script designed to be run from the scripts/operator_api/ directory


def getTosaArgTypes(tosaXml):
    """
    Returns a list of the TOSA argument types from tosa.xml.
    """
    argTypes = {
        "tensor_t",
        "in_t",
        "out_t",
        "mul_t",
        "weight_t",
        "in_out_t",
        "tensor_list_t",
    }
    argTypesXml = tosaXml.getElementsByTagName("type")
    for argTypeXml in argTypesXml:
        argTypes.add(argTypeXml.getAttribute("name"))
    argTypes.remove("TABLE_SIZE")
    return argTypes


def getTosaDataTypes(tosaXml):
    """
    Returns a list of the TOSA data types from tosa.xml.
    """
    argTypes = getTosaArgTypes(tosaXml)
    dataTypes = set()
    dataTypesXml = tosaXml.getElementsByTagName("typesupport")
    for dataTypeXml in dataTypesXml:
        for argType in argTypes:
            dataType = dataTypeXml.getAttribute(argType)
            if dataType != "":
                dataTypes.add(f"tosa_datatype_{dataType}")
    return sorted(dataTypes)


def getSerializeOpType(tosaOpName):
    """
    Returns the Serialization library operator that matches the TOSA operator specified.
    """
    map = {
        "avg_pool2d": "Pool",
        "conv2d": "Conv",
        "conv3d": "Conv",
        "depthwise_conv2d": "Conv",
        "fully_connected": "FullyConnected",
        "matmul": "MatMul",
        "max_pool2d": "Pool",
        "transpose_conv2d": "Conv",
        "clamp": "Clamp",
        "arithmetic_right_shift": "ArithmeticRightShift",
        "mul": "Mul",
        "table": "Table",
        "negate": "Negate",
        "pad": "Pad",
        "reshape": "Reshape",
        "slice": "Slice",
        "tile": "Tile",
        "transpose": "Transpose",
        "resize": "Resize",
        "rescale": "Rescale",
        "cond_if": "CondIf",
        "while_loop": "WhileLoop",
    }
    if tosaOpName not in map.keys():
        return "None"
    else:
        return map[tosaOpName]


def getSerialLibAttsForOp(tosaOpName, allSerialLibAtts, tosaArgs):
    """
    Returns the attributes required by the Serialization library for the TOSA operator specified.
    Generates code to initialize Serialization library attributes. If a matching TOSA argument exists,
    that value is used for initialization, otherwise a default value e.g. 0 is used.
    """
    serLibOpType = getSerializeOpType(tosaOpName)
    if serLibOpType not in allSerialLibAtts.keys():
        return {}
    else:
        serLibOpAtts = copy.deepcopy(allSerialLibAtts[serLibOpType])
        tosaArgsDict = {arg["name"]: arg for arg in tosaArgs}
        serTosaTypeMap = {"ResizeMode": "tosa_mode"}
        for att in serLibOpAtts:
            attName = att["name"]
            attType = att["dType"]
            init = ""
            # Translate TOSA data types to Serialization library data types for initialization
            if attType in serTosaTypeMap.keys():
                init = f"const {attType} {attName} = translate_client_{serTosaTypeMap[att['dType']]}(client_{attName});"
            # Initialize Serialization library attributes to their matching function parameter
            elif attName in tosaArgsDict:
                if att["SV"] == "V":
                    if tosaArgsDict[attName]["type"] == "tosa_tensor_t":
                        init = f"std::vector<{attType}> {attName};"
                        init = (
                            init
                            + f"size_t {attName}_size = client_{attName}.size / sizeof({attType});"
                        )
                        init = (
                            init
                            + f"{attType}* {attName}_data = reinterpret_cast<{attType}*>(client_{attName}.data);"
                        )
                        init = (
                            init
                            + f"{attName}.assign({attName}_data, {attName}_data + {attName}_size);"
                        )
                    else:
                        init = f"const std::vector<{attType}> {attName}"
                        shape = tosaArgsDict[attName]["shape"]
                        if shape == "[]":
                            init = (
                                init
                                + f"(&client_{attName}[0], &client_{attName}[0] + client_{attName}_len);"
                            )
                        else:
                            init = (
                                init
                                + f"(&client_{attName}[0], &client_{attName}{shape});"
                            )
                else:
                    init = ""
            else:
                # Initialize Serialization library attributes with no matching fuction parameter
                if att["SV"] == "V":
                    init = f"std::vector<int32_t> {attName};"
                else:
                    if att["dType"] == "DType":
                        att["dType"] = "tosa::DType"
                        init = f"const tosa::DType {attName} = tosa::DType::DType_FP32;"
                    else:
                        init = f"const {attType} {attName} = 0;"
            att["init"] = init
        return serLibOpAtts


def updateTosaArgs(tosaArgs, serialLibAtts, tosaXml):
    """
    Replace TOSA argument data types with their matching Serialization attribute data types.
    Delete TOSA arguments where the type couldn't be determined.
    Add Serialization attributes that have no matching TOSA argument.
    """
    tosaArgTypes = getTosaArgTypes(tosaXml)
    serAttsDict = {att["name"]: att for att in serialLibAtts}
    tosaArgsNames = [arg["name"] for arg in tosaArgs]
    delTosaArgs = []
    # Replace TOSA argument data types with their matching Serialization attribute data types.
    for tosaArg in tosaArgs:
        if tosaArg["type"] in tosaArgTypes:
            if tosaArg["name"] in serAttsDict:
                tosaArg["type"] = serAttsDict[tosaArg["name"]]["dType"]
            else:
                # Delete TOSA argument whose data type can't be determined
                delTosaArgs.append(tosaArgsNames.index(tosaArg["name"]))
                # Delete corresponding length argument if one exists
                lenArgName = f"{tosaArg['name']}_len"
                if lenArgName in tosaArgsNames:
                    delTosaArgs.append(tosaArgsNames.index(lenArgName))
    # Delete TOSA arguments where the type couldn't be determined
    for index in sorted(delTosaArgs, key=int, reverse=True):
        del tosaArgs[index]
    # Add Serialization attributes that have no matching TOSA argument
    tosaArgNames = [arg["name"] for arg in tosaArgs]
    for serAtt in serialLibAtts:
        attName = serAtt["name"]
        attType = serAtt["dType"]
        if (attName not in tosaArgNames) and (not attType == "tosa::DType"):
            serAttName = serAtt["name"]
            if serAtt["SV"] == "V":
                # For vector data types, insert a matching length argument
                tosaArgs.insert(
                    len(tosaArgs) - 1,
                    {
                        "name": f"{serAttName}_len",
                        "type": "int32_t",
                        "shape": "",
                        "category": "",
                    },
                )
                init = f"const std::vector<{attType}> {attName}(&client_{serAttName}[0], &client_{serAttName}[0] + client_{serAttName}_len);"
                shape = "[]"
            else:
                init = ""
                shape = ""
            serAtt["init"] = init
            # Insert new argument
            tosaArgs.insert(
                len(tosaArgs) - 1,
                {
                    "name": serAttName,
                    "type": serAtt["dType"],
                    "shape": shape,
                    "category": "",
                },
            )


def getOperators(tosaXml):
    """
    Return a list of TOSA operators as defined by tosa.xml.
    """
    operators = []
    ignoreOps = [
        "while_loop",
        "cond_if",
        "const",
        "custom",
        "fft2d",
        "rfft2d",
        "variable",
        "variable_read",
        "variable_write",
    ]
    opsXml = tosaXml.getElementsByTagName("operator")
    allSerialLibAtts = getSerialLibAtts()
    for opXml in opsXml:
        opName = opXml.getElementsByTagName("name")[0].firstChild.data.lower()
        if opName not in ignoreOps:
            operator = {"name": opName}
            operator["serializeAttType"] = getSerializeOpType(opName)
            tosaArgs = getTosaArgs(opXml)
            serialLibAtts = getSerialLibAttsForOp(opName, allSerialLibAtts, tosaArgs)
            # Handle "axis" arguments
            axisList = [arg["name"] for arg in tosaArgs if arg["name"] == "axis"]
            if operator["serializeAttType"] == "None" and len(axisList) > 0:
                operator["serializeAttType"] = "Axis"
                serialLibAtts = [
                    {
                        "name": "axis",
                        "dType": "int32_t",
                        "SV": "S",
                        "init": "",
                    }
                ]
            updateTosaArgs(tosaArgs, serialLibAtts, tosaXml)
            operator["arguments"] = tosaArgs
            operator["serialLibAtts"] = serialLibAtts
            serializationAttNames = [att["name"] for att in serialLibAtts]
            operator["inputs"] = [
                arg["name"]
                for arg in tosaArgs
                if arg["category"] == "input"
                and arg["name"] not in serializationAttNames
            ]
            operator["outputs"] = [
                arg["name"] for arg in tosaArgs if arg["category"] == "output"
            ]
            operators.append(operator)
    return operators


def getTosaArgs(opXml):
    """
    Return the arguments required for the TOSA operator specified.
    """
    arguments = []
    argsXml = opXml.getElementsByTagName("argument")
    tosaTensorTypes = getTosaArgTypes(tosaXml)
    tosaTypeMap = {"bool_t": "bool", "uint6_t": "uint8_t", "mode_t": "tosa_mode_t"}
    for xmlArg in argsXml:
        argName = xmlArg.getAttribute("name").lower()
        if xmlArg.getAttribute("tensor-element-type") == "resize_mode_t":
            argType = "tosa_mode_t"
        else:
            argType = xmlArg.getAttribute("type")
        argShape = xmlArg.getAttribute("shape")
        argCategory = xmlArg.getAttribute("category")
        # FullyConnected workaround
        if (argName == "weight" or argName == "bias") and (argCategory == "attribute"):
            argCategory = "input"
        # Update argument type
        if argType[-1:] == "*":
            argType = argType[:-1]
        if argCategory in ["input", "output"] and argType in tosaTensorTypes:
            argType = "tosa_tensor_t"
            argShape = ""
        if argType in tosaTypeMap:
            argType = tosaTypeMap[argType]
        # Add a length argument for arrays with unknown compile-time size
        if argShape != "" and argShape[0] == "[" and not argShape[1:-1].isnumeric():
            argShape = "[]"
            arguments.append(
                {
                    "name": f"{argName}_len",
                    "type": "int32_t",
                    "shape": "",
                    "category": "",
                }
            )
        elif argShape == "" or not argShape[0] == "[":
            argShape = ""
        # Append argument
        arguments.append(
            {
                "name": argName,
                "type": argType,
                "shape": argShape,
                "category": argCategory,
            }
        )
    return arguments


def clangFormat(filename):
    cmd = ["clang-format", "-i", filename]
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(cmd, stdout=devnull)


def getSerialLibAtts():
    """
    Parse attribute.def file and return a dictionary where the keys are Serialization library operator names.
    The values are the arguments required by each Serialization library operator.
    """
    serialLibAtts = {}
    with open("../../thirdparty/serialization_lib/include/attribute.def") as file:
        preamble = True
        inAtt = False
        opName = ""
        args = []
        for line in file:
            if preamble and not line[: len("DEF_ATTRIBUTE(")] == "DEF_ATTRIBUTE(":
                continue
            else:
                preamble = False
            line = line.lstrip().rstrip()
            if not inAtt and "DEF_ATTRIBUTE(" in line:
                opName = line[len("DEF_ATTRIBUTE(") : line.find(",")]
                inAtt = True
            elif inAtt:
                vals = line.split(",")
                argName = vals[2].lstrip().strip()
                if ")" in argName:
                    argName = argName[:-1]
                arg = {
                    "name": argName,
                    "dType": vals[0].lstrip().strip(),
                    "SV": vals[1].lstrip().strip(),
                }
                args.append(arg)
                if ")" in line:
                    serialLibAtts[opName] = args
                    opName = ""
                    args = []
                    inAtt = False
    return serialLibAtts


def renderTemplate(environment, dataTypes, operators, template, outfile):
    content = template.render(dataTypes=dataTypes, operators=operators)
    with open(outfile, mode="w", encoding="utf-8") as output:
        output.write(content)
        print(f"Created {outfile}")

    clangFormat(outfile)


def generate(environment, dataTypes, operators):
    # Generate include/operators.h
    template = environment.get_template("operators_h.j2")
    outfile = os.path.join("..", "..", "reference_model", "include", "operators.h")
    renderTemplate(environment, dataTypes, operators, template, outfile)

    # Generate src/operators.cc
    template = environment.get_template("operators_cc.j2")
    outfile = os.path.join("..", "..", "reference_model", "src", "operators.cc")
    renderTemplate(environment, dataTypes, operators, template, outfile)


def getSerializeOpTypeMap():
    """
    Utility function for generating the map used in getSerializeOpType()
    """
    import re

    allSerialLibAtts = getSerialLibAtts()
    serAtts = [
        re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        for name in allSerialLibAtts.keys()
    ]
    serAtts = sorted(serAtts, key=len, reverse=True)
    tosaXml = minidom.parse("../../thirdparty/specification/tosa.xml")
    opsXml = tosaXml.getElementsByTagName("operator")
    opNames = [
        op.getElementsByTagName("name")[0].firstChild.data.lower() for op in opsXml
    ]
    map = {}
    for opName in opNames:
        for serAtt in serAtts:
            if serAtt in opName:
                components = serAtt.split("_")
                map[opName] = "".join(x.title() for x in components)
    return map


if __name__ == "__main__":
    environment = Environment(loader=FileSystemLoader("templates/"))
    tosaXml = minidom.parse("../../thirdparty/specification/tosa.xml")
    dataTypes = getTosaDataTypes(tosaXml)
    operators = getOperators(tosaXml)
    generate(environment, dataTypes, operators)
