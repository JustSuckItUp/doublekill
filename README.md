# doublekill

## 1.get .onnx file
```
python c1.py
```
## 2. get trt engine file
step1.生成depoly.py
fp32:
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_poly_32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly.py" --trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256]
```
fp16:
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_poly_32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly.py" --trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256] --fp16
```
step2.生成engine
```
python depoly.py
```
## 3. compare speed:
使用polygraphy对比精度，运行以下文件：
```
python3 compare_speed.py
```
运行结果通过了精度校验。
