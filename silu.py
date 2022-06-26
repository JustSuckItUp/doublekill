import onnx
import onnx_graphsurgeon as gs
import numpy as np
graph = gs.import_onnx(onnx.load("./mobilevit.onnx"))
sigmoids =  [node for node in graph.nodes if node.op == 'Sigmoid']
for sig in sigmoids:
    silu_inputs = sig.inputs
    mul = sig.o()
    print(mul)
    mul_outputs = mul.outputs
    silu = gs.Node(op='SiLU',inputs=silu_inputs,outputs=mul_outputs)
    graph.nodes.append(silu)
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./mobilevit_silu.onnx")