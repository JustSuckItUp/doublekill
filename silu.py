import onnx
import onnx_graphsurgeon as gs
import numpy as np
graph = gs.import_onnx(onnx.load("./mobilevit.onnx"))
sigmoids =  [node for node in graph.nodes if node.op == 'Sigmoid']
for sig in sigmoids:
    silu_inputs = sig.inputs
    mul = sig.o()
    #print(mul)
    mul_outputs = mul.outputs
    silu = gs.Node(op='SiLU',inputs=silu_inputs,outputs=mul_outputs)
    graph.nodes.append(silu)
    print(mul.o())
    downstreams = [node for node in graph.nodes if mul_outputs[0] in node.inputs]
    for ds in downstreams:
        for i in range(len(ds.inputs)):
            if ds.inputs[i] == mul_outputs[0]:
                ds.inputs[i] = silu.outputs[0]
    graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./mobilevit_silu.onnx")