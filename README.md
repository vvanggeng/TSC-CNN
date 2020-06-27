# TSC-CNN

基于一维卷积神经网络（1D-CNN）的多元时间序列分类

## 项目背景

该项目为基于一维卷积神经网络的多元时间序列分类方法，实际问题被抽象为时间序列的分类问题，实际输入为4个传感器信号，分别对应16个类别，搭建1D-CNN然后训练网络对多元时间序列进行分类。



## 1D-CNN

无论是一维、二维还是三维，卷积神经网络（CNNs）都具有相同的特点和相同的处理方法。关键区别在于输入数据的维数以及特征检测器（或滤波器）如何在数据之间滑动，一维和二维CNN处理过程对比。

## 网络搭建

首先分析网络的输入输出，输入为包含4个时间序列的信号，长度为1050，即输入为（1050，4），而输出对应16种类别，输出为（，16）。实质上输入时先将单个训练数据的4个时间序列展平为（4200，1），传入网络后再reshape为（1050，4）。网络构建代码如下：

```python
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(TIME_PERIODS*num_sensors,)))
    model.add(Conv1D(16, 8,strides=2, activation='relu',input_shape=(TIME_PERIODS,num_sensors)))
    model.add(Conv1D(16, 8,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(Conv1D(512, 2,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
```

运行得到网络结构如下:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape_1 (Reshape)          (None, 1050, 4)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 522, 16)           528       
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 261, 16)           2064      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 130, 16)           0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 65, 64)            4160      
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 33, 64)            16448     
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 16, 64)            0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 8, 256)            65792     
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 4, 256)            262400    
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 2, 256)            0         
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 2, 512)            262656    
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 2, 512)            524800    
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 1, 512)            0         
_________________________________________________________________
global_average_pooling1d_1 ( (None, 512)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                8208      
=================================================================
Total params: 1,147,056
Trainable params: 1,147,056
Non-trainable params: 0
_________________________________________________________________
None
```

## 训练数据

所提供数据格式为二进制编码的bat文件，其内部数据格式为（1050，4），其中前两行为数据采集信息。包含1和2两个文件夹，每组都包含训练和验证数据。

数据下载见

[data](https://github.com/vvanggeng/TSC-KNN/tree/master/data)

## 其他问题

其他问题见代码，包含了详细的注释
