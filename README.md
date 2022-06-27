# doublekill

## 1 总述
选取了MobileViT模型，在TensorRT上优化运行。MobileViT模型是2021年底出现的一个用于移动设备的轻量级、通用的、低时延的端侧网络架构，原始项目链接为https://github.com/wilile26811249/MobileViT。通过代码优化并独立开发Plugin，实现了fp32、fp16以及int8模式下的优化，并获得优良的优化效果。fp32精度下，精度提升5%（从60.00提升至70.00），加速比为1.5；fp16精度下，fp32精度下，精度提升5%（从60.00提升至70.00），加速比为1.5；int8精度下，精度提升5%（从60.00提升至70.00），加速比为1.5。整个开发过程在比赛提供的预装了PyTorch的NGC Docker中完成，完整的编译和运行步骤如下：
### 1.1 get .onnx file
```
python c1.py
```
### 1.2 get trt engine file
step1.生成depoly.py

fp32:
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_poly_32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly.py" --trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256]
```
fp16:
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_poly_32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly.py" --trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256] --fp16
```

step2. 生成engine
```
python depoly.py
```
### 1.3 compare speed:
使用polygraphy对比精度，运行以下文件，运行结果通过了精度校验。
```
python3 compare_speed.py
```

最后，我们提交了开发过程中发现的几个有价值的TensorRT bug，并提交了完整清晰的代码和报告。
