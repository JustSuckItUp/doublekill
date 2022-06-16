import torch
import models
from time import time
import tensorrt as trt
import os
from cuda import cudart


trtfile = './mobilevit.plan'
onnxFile = './mobilevit.onnx'

img = torch.randn(1, 3, 256, 256,requires_grad=True)
net = models.MobileViT_S()
img = img.cuda()
net = net.cuda()
net = net.eval()

def torch_latency(times):
    for i in range(20):
        out = net(img)
    tik = time()
    for i in range(times):
        out = net(img)
    tok = time()
    torch_time_per_image = (tok-tik)/times
    return torch_time_per_image

def trt_excute():
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

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, 256, 256])
    _, stream = cudart.cudaStreamCreate()
    print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0))
    print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1))

trt_excute()