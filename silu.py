import onnx
import onnx_graphsurgeon as gs
import numpy as np
graph = gs.import_onnx(onnx.load("./mobilevit.onnx"))
sig = [node for node in graph.nodes if node.name == 'Sigmoid_32'][0]
p = sig.i()
c = sig.o()
g = c.o()
children = [node for node in graph.nodes if c.outputs[0] in node.inputs]

#print(p,c,g,sep='='*50)
print(children)
p.outputs = [gs.Variable(name='parent_output')]
silu_outputs = [gs.Variable(name='silu_output')]
silu = gs.Node(op='SiLU',inputs=p.outputs,outputs=silu_outputs)
graph.nodes.append(silu)
g.inputs[0] = silu_outputs[0]

# sigmoids =  [node for node in graph.nodes if node.op == 'Sigmoid']
# for sig in sigmoids:
#     parent_node = sig.i()
#     parent_node.outputs = [gs.Variable(name='parent_output_'+str(i))]
#     silu_inputs = parent_node.outputs
#     mul = sig.o()
#     #print(mul)
#     mul_outputs = mul.outputs
#     silu_outputs = [gs.Variable(name='silu_output_'+str(i))]
#     silu = gs.Node(op='SiLU',inputs=silu_inputs,outputs=silu_outputs)
#     graph.nodes.append(silu)
#     print(mul.o())
#
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./mobilevit_silu.onnx")