# doublekill

## 1 总述
选取了MobileViT模型，在TensorRT上优化运行。MobileViT模型是2021年底出现的一个用于移动设备的轻量级、通用的、低时延的端侧网络架构，原始项目链接为https://github.com/wilile26811249/MobileViT。 
**通过代码优化并独立开发Plugin，实现了fp32、fp16以及int8模式下的优化，并获得优良的优化效果。**

* fp32精度下，利用polygraphy对误差进行校验，并通过校验，fps提升29.43倍；
* fp16精度下，利用polygraphy对误差进行校验，并通过校验，fps提升51.52倍。
* INT8模式下没有通过polygraphy误差校验，FPS提升27.58倍。

**实现了SiLU的plugin，FP32，FP16加速比分别为24.88，30.48**
整个开发过程在比赛提供的预装了PyTorch的NGC Docker中完成，完整的编译和运行步骤如下：
### 1.1 get .onnx file
```
conda activate trt
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

**图中可得出结论：**

**1.FP32,FP16,INT8均比CPU以及torch自带gpu运行速度快，而且快特别多**

**2.FP16速度比FP32快**

**3.应用SiLU_plugin后速度变慢，猜测是自己生成的plugin没有官方算子快**

**4.INT8比FP32还慢，结合上述精度部分INT8也不合格，猜测是实现过程有问题**


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

## 3 模型选取的动机：
选取MobileViT的动机有以下几点：

1.MobileViT本身和trt加速是契合的，它们都可以对模型进行轻量化，从而让模型更易于应用到一些硬件平台。

2.目前并没有对MobileViT，尤其是MobileViT block的trt实现，这对一个效果很好的轻量化模型来说是很可惜的，此外MobileViT block是一个即插即用的模块，开发出对应的plugin会对之后其他的工作有很大的贡献。

## 4 优化过程

MobileViT项目已经开源了训练好的模型，接下来需要完成的是迁移到TensorRT中进行部署。

（1）搭建环境、跑通原项目代码；

（2）Pytorch框架导出ONNX，考虑到加速性能和精度的trade-off,尝试FP32、FP16、INT8三种精度的加速；

（3）使用trtexe、polygragh或者parser转化为TensorRT engine；

（4）针对MobileViT中激活函数silu,即x·sigmoid(x)，设计一个名为SiLU的plugin。
受到https://www.zhihu.com/question/539205443/answer/2543602804?utm_source=wechat_session&utm_medium=social&utm_oi=623078813148647424&utm_content=group2_Answer&utm_campaign=shareopn
中大佬的启发，我们从原理上寻找优化的思路


![image](https://user-images.githubusercontent.com/47712489/175877944-0f42fb0e-c6aa-4958-a2fb-f82b187b60ab.png) 

MobileViT模型采用了SiLU作为激活函数，对于SiLU激活函数，onnx里面会用 sigmoid+mul 的方式进行表示，tensorRT进行推理的时候会触发pointwise operator融合，把 sigmoid+mul 融合成一个 PWN 算子进行计算，但PWN算子不会进一步和前面的 Conv 进行融合。这导致对于上述子图，trt要启动两个kernel完成计算。而如果使用relu作为激活函数，relu与conv会融合，从而只需一个kernel完成所有运算。
我们以此作为切入点编写plugin进行优化，试图**通过写一个plugin,将sigmoid+mul,也就是PWN算子和前一步的Conv融合**，本次比赛我们只实现了PWN算子,即SiLU的plugin,但在我们实现的plugin的基础上，是有进一步扩展，实现我们想法的潜力的。在未来的工作计划中我们会进一步将conv层和sigmoid+mul融合为一个算子，编写plugin实现。算子融合能够减少访存和拷贝数据量，提高访问效率，这是一个非常不错的优化思路，由于比赛时间有限，我们暂未能实现这一部分，在之后我们将继续学习和探索，进一步补充完成。

本次比赛中，我们实现了：

**1.FP32、FP16、INT8的优化**

**2.SiLU的plugin实现,接入plugin后实现FP32,FP16**


具体的实现步骤如下：

**1.torch中生成mobilevit.onnx**

**2.利用polygraphy或者trtexec生成FP32、FP16的engine，并进行精度校验。**

**3.生成calibrator的cache和校验集**

**4.利用trtexec和第二步中生成文件生成INT8的engine**

**5.利用onnx库对mobilevit.onnx进行图优化，将Sigmoid+Mul层替换为SiLU层**



优化前计算图：


![image](https://user-images.githubusercontent.com/47239326/175954919-a54431b6-49dc-4384-b8df-445f2355bfd9.png)


优化后计算图：


![image](https://user-images.githubusercontent.com/47239326/175955071-e56d2ca9-c886-4a67-a34f-1d1fac9edba6.png)


**6.编写SiLU的plugin实现，并编译为.so**

**7.利用5.6步中生成文件和trtexec生成FP32、FP16的engine。**

**8.进行速度测算。**



## 5 精度与加速效果
### 5.1 软硬件环境

* 比赛提供的云计算节点，配置Ubuntu 20.04, NVIDIA A10
* trt环境：最新的TensorRT8.4.1.4版本（尚未对外发布）
* others: torch 1.8.1+cu111  torchvision 0.9.1+cu111  cuda-python 11.7.0

### 5.2 实验结果


|   | FPS | ratio |
| :------| ------: | :------: |
| CPU(torch) | 20.17 | 1 |
| GPU(torch) | 93.92 | 4.66 |
| FP32 | 593.69 | 29.43 |
| FP16 | 1039.15 | 51.52 |
| INT8 | 556.35 | 27.58 |
| FP32_SiLU | 501.79 | 24.88 |
| FP16_SiLU | 614.86 | 30.48 |

## 6 Bug报告

### 6.1 使用polygraphy生成FP32、FP16的engine并进行校验时，通过了校验，但是用onnxparser进行手动检查时torch输出和onnxparser输出对不上，即使onnxparser直接在运行期调用polygraphy生成并校验通过的engine，也不一致。


**bug未解决。。**

## 7.经验与体会
本次比赛从初赛到复赛，我们从小白逐渐开始学会了一点点trt部署的知识。复赛最后很多地方感觉都是小问题，但是碍于我们自己太菜了以及时间不太够，所以最后很多地方还是挺可惜的。不过总的来说还是收获了很多东西，希望NV关于trt的比赛越办越好！
