# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:26:13 2020

@author: wg
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import numpy as np
from sklearn import metrics
import seaborn as sns
import time

#start = time.time()

LABELS=[]#标签列表
for i in range(16):
    LABELS.append(chr(ord('A')+i))
                 
def show_confusion_matrix(validations, predictions):
    '''
    生成结果的混淆矩阵
    '''
    matrix = metrics.confusion_matrix(validations, predictions)  
    plt.figure('CNN',figsize=(10, 8))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                annot_kws={'size':14, 'color':'w'},
                fmt="d",)
    plt.title("Confusion Matrix of CNN_TSC",fontsize=18)
    plt.ylabel("True Label",fontsize=16)
    plt.xlabel("Predicted Label",fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=16)
    plt.show()
    
def load_file(filepath):
    '''
    读取指定文件返回为np数组形式
    '''
    f =open(file=filepath,mode='rb')
    data=[]
    for line in f.readlines()[2:]:
        num=[]
        for s in line.decode('UTF-8').replace('\n','').split('\t'):
            num.append(np.float(s))
        data.append(num)          
    f.close() 
    return np.array(data).T

def load_dataset(data_rootdir, dirname,istrain):
    '''
    遍历路径载入数据集
    '''
    filename_list = []
    filepath_list = []
    S=[]
   
    # 利用os.walk() 方法遍历文件、目录。
    for rootdir, dirnames, filenames in os.walk(data_rootdir + dirname):
        filenames.sort(key = lambda x: int(x[:-4]))#此步按文件名序号排序         
        for filename in filenames:
            filename_list.append(filename)
            filepath_list.append(os.path.join(rootdir, filename))
        #print(filename_list)
        #print(filepath_list)
                
    x=lable_creat(16,int(len(filename_list)/16),4,False)
    for i in range(len(filepath_list)):
        data=load_file(filepath_list[i])
        #加标签
        if istrain:
            data=np.column_stack((data,np.array(x[i]).T))
        S.append(data)
        
    #乱序
    if istrain:
        np.random.shuffle(S)
       
    S=np.vstack((S))        
    df=pd.DataFrame(S)
    return df

def lable_creat(kind_num,train_num,sensor_num,ispre):
    '''
    为原始数据添加标签
    '''
    x=[]
    z=[]
    for i in range(0,kind_num):
        for j in range(0,train_num):
            z.append(i)
            y=[]
            for k in range(0,sensor_num):
                y.append(i)  
            x.append(y)
    if ispre:
        x=z
    return x


Data_rootdir='C:/Users/wg/Desktop/TSC/data/2/'#数据路径

#把数据文件分批喂入网络训练，确定每次喂多少
Batch_size = 5

#Long代表总训练数据集数目，Lens代表其中用于训练网络的数据数目（7：3划分）
Long = 80   
Lens = 56

#把标签转成oneHot
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)

def xs_gen(path=Data_rootdir,batch_size = Batch_size,train=True,Lens=Lens*4):
    '''
    训练数据生成器
    '''
    img_list=load_dataset(path, 'train/',True)
    if train:
        img_list = np.array(img_list)[:Lens]
        print("Found %s train items."%(int(len(img_list)/4))) #len(img_list)/4 注意此处输入包含4个时间序列
        steps = math.ceil((len(img_list)/4) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = np.array(img_list)[Lens:]
        print("Found %s test items."%(int(len(img_list)/4)))
        steps = math.ceil((len(img_list)/4) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):

            batch_list = img_list[i * batch_size*4 : i * batch_size*4 + batch_size*4] #batch_size*4 此处注意
            #np.random.shuffle(batch_list)
            x = np.array([file for file in batch_list[:,0:-1]])
            y = np.array([convert2oneHot(label,16) for label in batch_list[:,-1]])
            batch_x = x.reshape(int(x.shape[0]/4),1050*4) #数据展平
            batch_y=[]
            for i in range(0,y.shape[0],4):
                batch_y.append(y[i,:])
            batch_y=np.vstack((batch_y))
            yield batch_x, batch_y
            
def ts_gen(path=Data_rootdir,batch_size = Batch_size):
    '''
    验证数据生成器
    '''
    img_list=load_dataset(path, 'test/',False)
    img_list = np.array(img_list)
    print("Found %s test items."%(int(len(img_list)/4)))
    steps = math.ceil((len(img_list)/4) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size*4 : i * batch_size*4 + batch_size*4]
            #np.random.shuffle(batch_list)
            x = np.array([file for file in batch_list])
            #batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])
            batch_x = x.reshape(int(x.shape[0]/4),1050*4)#数据展平
            yield batch_x
      
TIME_PERIODS = 1050 #数据长度
num_sensors=4 #每个输入包含4个时间序列

def build_model(num_classes=16):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(TIME_PERIODS*num_sensors,))) #输入维度调整
    model.add(Conv1D(16, 8,strides=2, activation='relu',input_shape=(TIME_PERIODS,num_sensors))) #输入维度设置
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
    model.add(Dropout(0.3)) #减小网络对数据微小变化的敏感性，提高泛化性能
    model.add(Dense(num_classes, activation='softmax'))
    return(model)  

Train = True
if __name__ == '__main__':
    if Train == True:
        '''
        训练网络
        '''
        train_iter = xs_gen()
        val_iter = xs_gen(train=False)
        
        #自动保存最佳网络
        ckpt = keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss', save_best_only=True,verbose=1)

        model = build_model()
        opt = Adam(0.0002)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
        print(model.summary())

        model.fit_generator(
            generator=train_iter,
            steps_per_epoch=Lens//Batch_size,
            epochs=50,
            initial_epoch=0,
            validation_data = val_iter,
            nb_val_samples = (Long - Lens)//Batch_size,
            callbacks=[ckpt],
            )
        model.save("finishModel.h5")
    else:
        '''
        测试网络
        '''
        test_iter = ts_gen()
        model = load_model("best_model.h5")
        pres = model.predict_generator(generator=test_iter,steps=math.ceil(48/Batch_size),verbose=1)
        print(pres.shape)
        ohpres = np.argmax(pres,axis=1)
        print(ohpres.shape)
        validations=lable_creat(16,3,4,True)
        show_confusion_matrix(validations, ohpres)
        
#end = time.time()
#print ('time cost',end-start,'s')      

    
