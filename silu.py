import onnx
import onnx_graphsurgeon as gs
import numpy as np
graph = gs.import_onnx(onnx.load("./mobilevit.onnx"))
sigmoids =  [node for node in graph.nodes if node.op == 'Sigmoid']
i = 0
for sig in sigmoids:
    i += 1
    parent_node = sig.i()
    parent_node.outputs = [gs.Variable(name='parent_output_'+str(i))]
    silu_inputs = parent_node.outputs
    mul = sig.o()
    #print(mul)
    mul_outputs = mul.outputs
    silu_outputs = [gs.Variable(name='silu_output_'+str(i))]
    silu = gs.Node(op='SiLU',inputs=silu_inputs,outputs=silu_outputs)
    graph.nodes.append(silu)
    print(mul.o())

    graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./mobilevit_silu.onnx")