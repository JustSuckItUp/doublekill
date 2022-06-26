#!/usr/bin/env python3
# Template auto-generated by polygraphy [v0.36.2] on 06/26/22 at 17:11:17
# Generation Command: /usr/local/bin/polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_poly_32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script ./depoly.py --trt-min-shapes modelInput:[1,3,256,256] --trt-opt-shapes modelInput:[16,3,256,256] --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256] --plugins /root/workdir/plugins/siluPlugin/SiLU.so
# This script compares /root/workdir/doublekill/mobilevit.onnx between ONNX Runtime and TensorRT.

from polygraphy.logger import G_LOGGER
G_LOGGER.severity = G_LOGGER.VERBOSE

from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import CreateConfig as CreateTrtConfig, EngineFromNetwork, LoadPlugins, NetworkFromOnnxPath, Profile, SaveEngine, TrtRunner
from polygraphy.common import TensorMetadata
from polygraphy.comparator import Comparator, CompareFunc, DataLoader
import sys

# Data Loader
data_loader = DataLoader(input_metadata=TensorMetadata().add('modelInput', None, (1, 3, 256, 256)))

# Loaders
build_onnxrt_session = SessionFromOnnx('/root/workdir/doublekill/mobilevit.onnx')
load_plugins = LoadPlugins(plugins=['/root/workdir/plugins/siluPlugin/SiLU.so'], obj='/root/workdir/doublekill/mobilevit.onnx')
parse_network_from_onnx = NetworkFromOnnxPath(load_plugins)
profiles = [
    Profile().add('modelInput', min=[1, 3, 256, 256], opt=[16, 3, 256, 256], max=[32, 3, 256, 256])
]
create_trt_config = CreateTrtConfig(max_workspace_size=23622320128, profiles=profiles)
load_plugins_1 = LoadPlugins(plugins=['/root/workdir/plugins/siluPlugin/SiLU.so'], obj=parse_network_from_onnx)
build_engine = EngineFromNetwork(load_plugins_1, config=create_trt_config)
save_engine = SaveEngine(build_engine, path='mobilevit_poly_32.plan')

# Runners
runners = [
    OnnxrtRunner(build_onnxrt_session),
    TrtRunner(save_engine),
]

# Runner Execution
results = Comparator.run(runners, data_loader=data_loader)

success = True
# Accuracy Comparison
compare_func = CompareFunc.simple(rtol={'': 0.001}, atol={'': 0.001})
success &= bool(Comparator.compare_accuracy(results, compare_func=compare_func))

# Report Results
cmd_run = ' '.join(sys.argv)
if not success:
    G_LOGGER.critical("FAILED | Command: {}".format(cmd_run))
G_LOGGER.finish("PASSED | Command: {}".format(cmd_run))

