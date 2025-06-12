# TOSA Reference Model Graph Execution Example

This example demonstrates how the TOSA reference model internally builds and executes computational graphs.

## Overview

The example creates a simple graph:
```
input1 ──> ABS ──> temp ──┐
                          ├──> ADD ──> output
input2 ───────────────────┘
```

Input values:
- input1: [-2.0, -1.0, 1.0, 2.0]
- input2: [1.0, 1.0, 1.0, 1.0]

Expected output: [3.0, 2.0, 2.0, 3.0] (abs(input1) + input2)

## Key Components

### 1. **SubgraphTraverser** (`subgraph_traverser.cpp`)
The central scheduler that:
- Builds the graph from serialized TOSA operations
- Maintains a ready-list of operations that can execute
- Manages tensor allocation and lifetime
- Executes nodes in dependency order

### 2. **GraphNode** (`graph_node.cpp`)
Base class for all operations:
- Wraps the actual compute implementation
- Tracks input/output tensors
- Validates tensor attributes before execution
- Calls the underlying operator's `eval()` method

### 3. **Operators** (`ops/` directory)
Actual compute implementations:
- Template-based for different ranks and data types
- Use Eigen for tensor operations
- Example: `OpAbs<Rank, Dtype>` computes element-wise absolute value

### 4. **Tensor** (`tensor.cpp`)
Data containers:
- Owns the actual data buffer
- Tracks shape, data type, and validity
- Supports producer/consumer relationships

## Execution Flow

1. **Graph Construction**
   ```cpp
   SubgraphTraverser sgt(&block, nullptr, nullptr);
   sgt.initializeGraph();  // Creates GraphNode objects from serialized ops
   ```

2. **Tensor Linking**
   ```cpp
   sgt.linkTensorsAndNodes();  // Connects tensors to their producers/consumers
   ```

3. **Scheduling**
   - Nodes with all inputs ready are added to `nextNodeList`
   - Initially: only nodes with constant inputs (or graph inputs)

4. **Execution Loop**
   ```cpp
   while (!sgt.isFullyEvaluated()) {
       sgt.evaluateNextNode();  // Pops from ready-list, executes, updates consumers
   }
   ```

5. **Node Evaluation**
   - `GraphNode::eval()` calls operator's `init()` then `cpu_eval()`
   - Operator reads input tensors, computes, writes output tensors
   - Marks outputs as valid, triggering dependent nodes

## Building and Running

```bash
# First build the reference model
mkdir build && cd build
cmake .. && make

# Then build the example
cd ..
chmod +x reference_model/src/build_test_graph.sh
./reference_model/src/build_test_graph.sh

# Run the example
./test_graph_example
```

## Expected Output

```
=== TOSA Reference Model Graph Execution Demo ===

--- Initializing graph ---
Creating operator id_000,      ABS, 1 input tensors, 1 output tensors
Creating operator id_001,      ADD, 2 input tensors, 1 output tensors

--- Allocating tensors ---
Allocating tensor input1
Allocating tensor input2

--- Executing graph ---

Step 0: Next node list
Node type: ABS ID: 0 Eval Count: 0 On next node list: 1 Evaluated: 0
Done.
Evaluating node_000,      ABS, output tensor=temp

Step 1: Next node list
Node type: ADD ID: 1 Eval Count: 0 On next node list: 1 Evaluated: 0
Done.
Evaluating node_001,      ADD, output tensor=output

--- Results ---
Output tensor values: [3, 2, 2, 3]

Expected: [3.0, 2.0, 2.0, 3.0] (abs(input1) + input2)
```

## Debugging Tips

1. **Trace Execution**: Set `DEBUG_ENABLED(DEBUG_VERB_HIGH, GT)` to see detailed traces
2. **Dump Tensors**: Use `tensor->dumpTensor(FILE*)` to inspect values
3. **Graph Visualization**: `sgt.dumpGraph(FILE*)` shows the full graph structure
4. **Step Through**: Use debugger breakpoints in `evaluateNextNode()` to watch scheduling

## Extending the Example

To add more operations:

1. Add tensor definitions:
   ```cpp
   tensors.push_back(createSerTensor("new_temp", DType_FP32, shape));
   ```

2. Add operator:
   ```cpp
   operators.push_back(createSerOperator(Op_MUL, nullptr, {"temp", "new_input"}, {"new_temp"}));
   ```

3. Update graph inputs/outputs as needed

## Common Operations

- **Unary**: ABS, NEG, EXP, LOG, CEIL, FLOOR, etc.
- **Binary**: ADD, SUB, MUL, DIV, POW, MIN, MAX, etc.
- **Reduce**: REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN, etc.
- **Data layout**: RESHAPE, TRANSPOSE, PAD, SLICE, etc.
- **Neural network**: CONV2D, MATMUL, MAXPOOL2D, etc. 