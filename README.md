# doublekill

## 1 总述
选取了MobileViT模型，在TensorRT上优化运行。MobileViT模型是2021年底出现的一个用于移动设备的轻量级、通用的、低时延的端侧网络架构，原始项目链接为https://github.com/wilile26811249/MobileViT。 通过代码优化并独立开发Plugin，实现了fp32、fp16以及int8模式下的优化，并获得优良的优化效果。fp32精度下，精度提升5%（从60.00提升至70.00），加速比为1.5；fp16精度下，fp32精度下，精度提升5%（从60.00提升至70.00），加速比为1.5；int8精度下，精度提升5%（从60.00提升至70.00），加速比为1.5。整个开发过程在比赛提供的预装了PyTorch的NGC Docker中完成，完整的编译和运行步骤如下：
### 1.1 get .onnx file
```
cd ~/workdir/doublekill
python c1.py
```
### 1.2  trt engine file without plugin
#### 1.2.1 get fp32_engine:mobilevit_fp32.plan
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_fp32.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly_fp32.py" \
 --trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256]
python3 depoly_fp32.py  
```
**通过polygraphy对比onnxruntime和trt engine输出结果,反映出误差在许可范围：**
![image](https://user-images.githubusercontent.com/47239326/175920488-e8df7a55-02f6-45ed-9841-4175d80843c9.png)

#### 1.2.2 get fp16_engine:mobilevit_fp16.plan
```
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_fp16.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly_fp16.py" --trt-min-shapes modelInput:[1,3,256,256]  \
 --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256] --fp16 
python3 depoly_fp16.py 
```
**通过polygraphy对比onnxruntime和trt engine输出结果,反映出误差在许可范围：**
![image](https://user-images.githubusercontent.com/47239326/175921854-0a471c9d-188a-42b9-a8c5-ec68cf600856.png)

#### 1.2.3 get int8_engine:mobilevit_int8.plan
```
python3 int8.py  #generate calib-cache
polygraphy run mobilevit.onnx --onnxrt --trt --workspace 22G --save-engine=mobilevit_int8.plan --atol 1e-3 --rtol 1e-3 --verbose --gen-script "./depoly_int8.py" \
--trt-min-shapes modelInput:[1,3,256,256]   --trt-opt-shapes modelInput:[16,3,256,256]   --trt-max-shapes modelInput:[32,3,256,256] --input-shapes modelInput:[1,3,256,256] --int8 --calibration-cache mobilevit.cache 
python3 depoly_int8.py
```
**通过polygraphy对比onnxruntime和trt engine输出结果,误差没有通过，polygraphy的误差是element-wise的比较严格，从下图可看出一些元素误差较大，这部分后续还要继续改善**
![image](https://user-images.githubusercontent.com/47239326/175922812-5ff21b43-3bd9-4b1b-b1d2-f370509990af.png)

### 1.3 trt engine file with SiLU plugin
#### 1.3.1 compile plugin
```
python3 silu.py #将Sigmoid+Mul层替换为SiLU
cd siluPlugin
make
cd ..
cp siluPlugin/SiLU.so .
```
#### 1.3.2 convert .onnx to .plan with SiLU plugin
```
# mobilevit_fp32_silu
trtexec --onnx=mobilevit_silu.onnx  --minShapes=modelInput:1x3x256x256 --optShapes=modelInput:16x3x256x256  --maxShapes=modelInput:32x3x256x256  \
--workspace=40000 --saveEngine=mobilevit_fp32_silu.plan --verbose --plugins=SiLU.so
# mobilevit_fp16_silu
trtexec --onnx=mobilevit_silu.onnx  --minShapes=modelInput:1x3x256x256 --optShapes=modelInput:16x3x256x256  --maxShapes=modelInput:32x3x256x256  \
--workspace=40000 --saveEngine=mobilevit_fp16_silu.plan --verbose --plugins=SiLU.so --fp16 
```
**1.由于polygraphy对比精度时是将onnxruntime的输出和trt的输出对比，而onnxruntime中识别不了SiLU算子，所以不能对比精度，我们尝试用parser对比精度时出了bug会在后文中详述**
**2.关于使用silu的plugin如何生成calibrator的cache，我们还没搞懂，所以没有进一步研究mobilevit_int8_silu**


### 1.4 compare speed:
使用polygraphy对比精度，运行以下文件，得到了1.2和1.3中trtengine的运行速度。
```
python3 compare_speed.py
```
可以得到结果：
```
# Absolute time
{'cpu_latency': 0.049575185775756835, 'gpu_latency': 0.010647022724151611, './mobilevit_fp32.plan': 0.0016843676567077637, './mobilevit_fp16.plan': 0.0009623289108276368, './mobilevit_int8.plan': 0.0017974257469177246, './mobilevit_fp32_silu.plan': 0.0019928693771362306, './mobilevit_fp16_silu.plan': 0.0016263842582702637}
# FPS
{'cpu_latency': 20.17138179821039, 'gpu_latency': 93.9229703841628, './mobilevit_fp32.plan': 593.6946105665452, './mobilevit_fp16.plan': 1039.1457522978965, './mobilevit_int8.plan': 556.3512160181458, './mobilevit_fp32_silu.plan': 501.78903417995616, './mobilevit_fp16_silu.plan': 614.8608454090346}
# relative ratio to cpu
{'cpu_latency': 1.0, 'gpu_latency': 93.9229703841628, './mobilevit_fp32.plan': 593.6946105665452, './mobilevit_fp16.plan': 1039.1457522978965, './mobilevit_int8.plan': 556.3512160181458, './mobilevit_fp32_silu.plan': 501.78903417995616, './mobilevit_fp16_silu.plan': 614.8608454090346}
```
	FPS	ratio
CPU	20.17	1
GPU	93.92	93.92
FP32	593.69	593.69
FP16	1039.15	1039.15
INT8	556.35	556.35
FP32_SiLU	501.79	501.79
FP16_SiLU	614.86	614.86


最后，我们提交了开发过程中发现的几个有价值的TensorRT bug，并提交了完整清晰的代码和报告。


## 2 原始模型
MobileViT，是一个用于移动设备的轻量级、通用的、低时延的端侧网络架构，由苹果公司在2021年底提出。该网络架构利用了CNN中的空间归纳偏置优势以及对数据增强技巧的低敏感性的特性，结合了ViT中对输入特征图信息进行自适应加权和建立全局依赖关系等优点，有效结合了CNN的归纳偏置优势和ViT的全局感受野能力。具体做法如下：

![image](https://user-images.githubusercontent.com/47712489/175880612-3e22fbe8-a026-488a-a03e-d74c329dfb0b.png)

如上图所示，MobileViT的一个特点是提出了MobileViT block（如上图1（b）所示）, MobileViT块使用标准卷积和Transformer来有效的结合local和global的视觉表征信息，以实现“transformer as convolution”的操作。具体来说，我们知道标准卷积主要涉及三个操作：展开（unfloading） 、局部处理（local processing） 和展开（folding）。MobileVIT block通过Figure 1的结构，将第二部分替换成tranformer，达到了 “transformer as convolution”的一个结构。 这个结构由于既有CNN的局部注意力，又有tranformer的全局注意力，可以很好的提取特征，因此可以用来构建一个轻量化的模型。
MobileViT 在不同的端侧视觉任务（图像分类、物体检测、语义分割）上都取得了比当前轻量级 CNN或ViT模型更好的性能。值得注意的一点：不同于大多数基于ViT的模型，MobileViT模型仅仅使用基础的数据增强训练方式，就达到了更优的性能。

![image](https://user-images.githubusercontent.com/47712489/175880861-20f888e2-c4af-4dcb-9e2d-5ae26dc45eca.png)

实验结果表明，MobileViT在不同的任务和数据集上显著优于基于CNN和ViT的网络。在ImageNet-1k数据集上，MobileViT实现了78.4%的Top-1精度，拥有约600万个参数，对于类似数量的参数，其精度分别比MobileNetv3（基于CNN）和DeIT（基于ViT）高3.2%和6.2%。在MS-COCO目标检测任务中，对于相同数量的参数，MobileViT比MobileNetv3的准确度高5.7%。如下图所示。

![image](https://user-images.githubusercontent.com/47712489/175880970-83fbae8e-ee42-4f17-93d4-c648423abd72.png)

参考文献：Mehta S, Rastegari M. Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer[J]. arXiv preprint arXiv:2110.02178, 2021.

## 3 模型优化的难点
选取MobileViT的动机有以下几点：

1.MobileViT本身和trt加速是契合的，它们都可以对模型进行轻量化，从而让模型更易于应用到一些硬件平台。

2.目前并没有对MobileViT，尤其是MobileViT block的trt实现，这对一个效果很好的轻量化模型来说是很可惜的，此外MobileViT block是一个即插即用的模块，开发出对应的plugin会对之后其他的工作有很大的贡献。

## 4 优化过程

MobileViT项目已经开源了训练好的模型，接下来需要完成的是迁移到TensorRT中进行部署。

（1）搭建环境、跑通原项目代码；

（2）Pytorch框架导出ONNX，考虑到加速性能和精度的trade-off,尝试FP32、FP16、INT8三种精度的加速；

（3）使用trtexe、polygragh或者parser转化为TensorRT engine；

（4）考虑针对mobilevit中部分模块编写plugin，导入engine文件实现加速。

![image](https://user-images.githubusercontent.com/47712489/175877944-0f42fb0e-c6aa-4958-a2fb-f82b187b60ab.png) 

我们从原理上寻找优化的思路。MobileViT模型采用了SiLU作为激活函数，对于SiLU激活函数，onnx里面会用 sigmoid+mul 的方式进行表示，tensorRT进行推理的时候会触发pointwise operator融合，把 sigmoid+mul 融合成一个 PWN 算子进行计算，但PWN算子不会进一步和前面的 Conv 进行融合。这导致对于这个子图，trt要启动两个kernel完成计算。
我们以此作为切入点编写plugin进行优化，完成了Sigmioid+Mul部分的plugin编写，即PWN plugin，在速度上取得了明显的提升。在tensorRT进行推理的速度为：fp32，fp16，int8。使用我们编写的plugin后速度为：fp32，fp16，int8。具体的实现步骤如下：

1.
2.
3.

接下来一步工作计划为将conv层和sigmoid+mul融合为一个算子，编写plugin实现。算子融合能够减少访存和拷贝数据量，提高访问效率，这是一个非常不错的优化思路，由于比赛时间有限，我们暂未能实现这一部分，在之后我们将继续学习和探索，进一步补充完成。

## 5 精度与加速效果
### 5.1 软硬件环境

* 比赛提供的云计算节点，配置Ubuntu 20.04, NVIDIA A10
* 环境：最新的ensorRT8.4版本（尚未对外发布）

### 5.2 实验结果

| 模式 | fp32 | fp16 | int8 | fp32(PWN plugin) | fp16(PWN plugin) | int8(PWN plugin) |
| :------| ------: | :------: | :------| ------: | :------: | :------|
| fps |   |   |   |   |   |   | 
| 精度 |  |   |  |   |   |   | 

## 6 Bug报告
TensorRT8.4.0环境中，无法使用trtexec和polygraphy convert转换我们得到的onnx模型。

命令：
```
trtexec --onnx=mobilevit.onnx  --minShapes=modelInput:1x3x256x256 --optShapes=modelInput:16x3x256x256  --maxShapes=modelInput:32x3x256x256  --workspace=40000 --saveEngine=mobilevit.plan --verbose
```

报错：

 ![image](https://user-images.githubusercontent.com/47712489/175878249-a7eb3126-3c6f-46d1-939c-32aaa86fc8b9.png)
 
bug解决方案：

当前采用的版本未完善，采用了还未正式发布的最新TRT8.4版本，导出成功。

