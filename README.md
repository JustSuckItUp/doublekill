# doublekill

## 1.get .onnx file
```
python c1.py
```
## 2. get trt engine file

```
trtexec --onnx=mobilevit.onnx  --minShapes=modelInput:1x3x256x256 --optShapes=modelInput:16x3x256x256 \
 --maxShapes=modelInput:32x3x256x256  --workspace=40000 --saveEngine=mobilevit.plan --verbose
```
## 3. compare speed:
```
python3 compare_speed.py
```
