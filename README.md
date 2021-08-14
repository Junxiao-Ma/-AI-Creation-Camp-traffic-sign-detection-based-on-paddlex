# [AI创造营]基于PaddleX实现交通路标检测

      本项目用paddleX实现了对交通路标的检测

# 一、项目背景
 
     现在道路上车辆越来越多，好多驾驶人越来越不注意行车安全，也不注意看交通路标，于是导致了交通危险事故的发生。
     本项目就是解决驾驶人由于疲劳等原因忽视交通路标的问题。

# 二、数据集简介

本项目使用的数据集是：[[AI训练营]目标检测数据集合集](https://aistudio.baidu.com/aistudio/datasetdetail/103743)，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**

## 1.数据加载

```python
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```
解压完成后，左侧文件夹处会多一个名为objDataset的文件夹，该文件夹下有5个子文件夹：

barricade——Gazebo锥桶检测
facemask——口罩检测
fire——火焰检测
MidAutumn——中秋元素检测
roadsign_voc——交通路标检测
每个子文件夹下有2个文件夹，分别存放着图像（JPEGImages）和标注文件（Annotations），如下所示：

## 2.数据预处理
本项目使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要安装PaddleX。

```python
# 安装PaddleX
!pip install paddlex
```
使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。
# 划分数据集
```python
!paddlex --split_dataset --format VOC --dataset_dir objDataset/roadsign_voc --val_value 0.2 --test_value 0.1
```
# 数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)
```python
from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomDistort()
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomDistort()
])
```
读取PascalVOC格式的检测数据集，并对样本进行相应的处理。
```python
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/roadsign_voc',
    file_list='objDataset/roadsign_voc/train_list.txt',
    label_list='objDataset/roadsign_voc/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/roadsign_voc',
    file_list='objDataset/roadsign_voc/val_list.txt',
    label_list='objDataset/roadsign_voc/labels.txt',
    transforms=eval_transforms)
```
需要注意的是：
- **data_dir** (str): 数据集所在的目录路径。
- **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
- **label_list** (str): 描述数据集包含的类别信息文件路径。

需要将第二步数据准备时生成的labels.txt, train_list.txt, val_list.txt和test_list.txt配置到以上变量中


## 3.数据集查看


```python
# 查看数据集文件结构
!tree objDataset -L 2
```
objDataset  
├── barricade  
│   ├── Annotations  
│   └── JPEGImages  
├── facemask  
│   ├── Annotations  
│   └── JPEGImages  
├── fire  
│   ├── Annotations  
│   └── JPEGImages  
├── MidAutumn  
│   ├── Annotations  
│   └── JPEGImages  
└── roadsign_voc  
    ├── Annotations  
    └── JPEGImages

   15 directories, 0 files


# 三、模型选择和开发
PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本项目以骨干网络为DarkNet53的YOLOv3算法为例。



## 1.模型组网

### 先来简单给大家介绍一下YOLOv3中的DarkNet53
![](https://ai-studio-static-online.cdn.bcebos.com/aa2aae3bb8514bf591c2c83cbd49a47a643d31c0d7b14c9f80308e612b490938)
![](https://ai-studio-static-online.cdn.bcebos.com/ced4d0c382b7477493668aad67bf007342447f7b7e9b4f9a91cdeb8277b56ce1)
![](https://ai-studio-static-online.cdn.bcebos.com/77409f3bb1f648c180e6f56c6922b21fa75f322dbfa04a75ad20dbe745625527)





```python
# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 512),    # 隐层：线性变换层
    paddle.nn.ReLU(),              # 激活函数
    paddle.nn.Linear(512, 10)      # 输出层
)
```

## 2.加载目标检测模型
```python
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='DarkNet53')
```
## 3.说明模型需要的超参，然后就可以开始训练啦
```python
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')
```



```python
# 模型封装
model = paddle.Model(network)

# 模型可视化
model.summary((1, 28, 28))
```

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #    
    ===========================================================================
       Flatten-1       [[1, 28, 28]]           [1, 784]              0       
       Linear-1          [[1, 784]]            [1, 512]           401,920    
        ReLU-1           [[1, 512]]            [1, 512]              0       
       Linear-2          [[1, 512]]            [1, 10]             5,130     
    ===========================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    ---------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.01
    Params size (MB): 1.55
    Estimated Total Size (MB): 1.57
    ---------------------------------------------------------------------------

    {'total_params': 407050, 'trainable_params': 407050}

## 3.模型训练


```python
# 配置优化器、损失函数、评估指标
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=network.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
              
# 启动模型全流程训练
model.fit(train_dataset,  # 训练数据集
          eval_dataset,   # 评估数据集
          epochs=5,       # 训练的总轮次
          batch_size=64,  # 训练使用的批大小
          verbose=1)      # 日志展示形式
```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/5
    step 938/938 [==============================] - loss: 0.0325 - acc: 0.9902 - 7ms/step           
    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 157/157 [==============================] - loss: 7.0694e-04 - acc: 0.9807 - 6ms/step     
    Eval samples: 10000
    


## 4.模型评估测试


```python
# 模型评估，根据prepare接口配置的loss和metric进行返回
result = model.evaluate(eval_dataset, verbose=1)

print(result)
```

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 10000/10000 [==============================] - loss: 0.0000e+00 - acc: 0.9795 - 2ms/step         
    Eval samples: 10000
    {'loss': [0.0], 'acc': 0.9795}


## 5.模型预测

### 5.1 批量预测

使用model.predict接口来完成对大量数据集的批量预测。


```python
# 进行预测操作
result = model.predict(eval_dataset)

# 定义画图方法
def show_img(img, predict):
    plt.figure()
    plt.title('predict: {}'.format(predict))
    plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
    plt.show()

# 抽样展示
indexs = [2, 15, 38, 211]

for idx in indexs:
    show_img(eval_dataset[idx][0], np.argmax(result[0][idx]))
```

    Predict begin...
    step 10000/10000 [==============================] - 1ms/step        
    Predict samples: 10000


### 5.2 单张图片预测

采用model.predict_batch来进行单张或少量多张图片的预测。


```python
# 读取单张图片
image = eval_dataset[501][0]

# 单张图片预测
result = model.predict_batch([image])

# 可视化结果
show_img(image, np.argmax(result))
```

# 四、效果展示

说明你的项目应该如何去运行。

并简单说明你的项目取得了哪些成果，效果如何。最好附上图片。

# 五、总结与升华

本项目在模型预测和评估方面还没有完成，16日完成之前会更新

# 个人简介

[马骏骁](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/824948)

<https://aistudio.baidu.com/aistudio/personalcenter/thirdview/824948>