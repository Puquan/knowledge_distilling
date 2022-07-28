import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

for csv_file in os.listdir('./data_record/resnet_vgg'):
    df = pd.read_csv('./data_record/resnet_vgg/' + csv_file)
    column = df.iloc[3]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Accuracy") # 设置纵轴名称
plt.title('Resnet18 Knowledge Distillation with VGG16(Test)')
plt.legend() #显示图例
plt.savefig("./Accuracy_test.jpg")
plt.clf()

for csv_file in os.listdir('./data_record/resnet_vgg'):
    df = pd.read_csv('./data_record/resnet_vgg/' + csv_file)
    column = df.iloc[2]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])

plt.xlabel('epoch')
plt.ylabel("Loss") # 设置纵轴名称
plt.title('Resnet18 Knowledge Distillation with VGG16(Test)')
plt.legend() #显示图例
plt.savefig("./Loss_test.jpg")
plt.clf()


for csv_file in os.listdir('./data_record/resnet_vgg'):
    df = pd.read_csv('./data_record/resnet_vgg/' + csv_file)
    column = df.iloc[1]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Accuracy") # 设置纵轴名称
plt.title('Resnet18 Knowledge Distillation with VGG16(Train)')
plt.legend() #显示图例
plt.savefig("./Accuracy_train.jpg")
plt.clf()



for csv_file in os.listdir('./data_record/resnet_vgg'):
    df = pd.read_csv('./data_record/resnet_vgg/' + csv_file)
    column = df.iloc[0]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Loss") # 设置纵轴名称
plt.title('Resnet18 Knowledge Distillation with VGG16(Train)')
plt.legend() #显示图例
plt.savefig("./Loss_train.jpg")
plt.clf()