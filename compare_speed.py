import torch
import models
from time import time
import numpy as np
import os
from cuda import cudart
import tensorrt as trt

def distance(a,b):
    l2_distance = np.linalg.norm(a-b)
    cos_distance = 1-np.dot(a,b.T).squeeze(0).squeeze(0)/(np.linalg.norm(a)*(np.linalg.norm(b)))
    return [l2_distance,cos_distance]


trtfile = './mobilevit.plan'
onnxFile = './mobilevit.onnx'
nRound = 20

img = torch.randn(1, 3, 256, 256,dtype=torch.float32,requires_grad=False)
net = models.MobileViT_S()
net.cpu()
# img = img.cuda()
# net = net.cuda()
net = net.eval()

out_gpu = None
out_cpu = None
#cpu
for i in range(20):
    out_cpu = net(img)
tic = time()
for i in range(nRound):
    out_cpu = net(img)
toc = time()
cpu_latency = (toc - tic) / nRound
out_cpu = out_cpu.detach().numpy()
#gpu
img = img.cuda()
net = net.cuda()
for i in range(20):
    out_gpu = net(img)

torch.cuda.synchronize()
tic = time()
for i in range(nRound):
    out_gpu = net(img)
torch.cuda.synchronize()
toc = time()
gpu_latency = (toc - tic) / nRound
out_gpu = out_gpu.cpu().detach().numpy()
g2c_l2,g2c_cos = distance(out_cpu,out_gpu)
print(g2c_l2,g2c_cos)


#
logger = trt.Logger(trt.Logger.VERBOSE)
if os.path.isfile(trtfile):
    with open(trtfile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.flags = 1 << int(trt.BuilderFlag.FP16)
    config.max_workspace_size = 3 << 30
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing ONNX file!")

    input_name = 'modelInput'
    input_shape = [-1,3,256,256]
    profile.set_shape(input_name,(1,3,256,256),(16,3,256,256),(32,3,256,256))
    config.add_optimization_profile(profile)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtfile, 'wb') as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

#运行期
context = engine.create_execution_context()
context.set_binding_shape(0, [1, 3, 256, 256])
_, stream = cudart.cudaStreamCreate()
print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0))
print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1))
inputH0 = np.ascontiguousarray(img.cpu().numpy().reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                       stream)
for i in range(20):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaStreamSynchronize(stream)
tic = time()
for i in range(nRound):
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaStreamSynchronize(stream)
toc = time()
trt_latency = (toc-tic)/nRound
print(cpu_latency,gpu_latency,trt_latency)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes,
                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

# print("outputH0:", outputH0.shape)
# print(outputH0)
cudart.cudaStreamSynchronize(stream)
cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)

print("Succeeded running model in TensorRT!")
t2c_l2,t2c_cos = distance(out_cpu,outputH0)
print(t2c_l2,t2c_cos)

