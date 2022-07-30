import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

for csv_file in os.listdir('./data_record/resnet18_vgg16'):
    df = pd.read_csv('./data_record/resnet18_vgg16/' + csv_file)
    column = df.iloc[3]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Accuracy") 
plt.title('Resnet18 Knowledge Distillation with VGG16(Test)')
plt.legend() 
plt.savefig("./figures/resnet_vgg/Accuracy_test.jpg")
plt.clf()

for csv_file in os.listdir('./data_record/resnet18_vgg16'):
    df = pd.read_csv('./data_record/resnet18_vgg16/' + csv_file)
    column = df.iloc[2]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])

plt.xlabel('epoch')
plt.ylabel("Loss") 
plt.title('Resnet18 Knowledge Distillation with VGG16(Test)')
plt.legend() 
plt.savefig("./figures/resnet_vgg/Loss_test.jpg")
plt.clf()


for csv_file in os.listdir('./data_record/resnet18_vgg16'):
    df = pd.read_csv('./data_record/resnet18_vgg16/' + csv_file)
    column = df.iloc[1]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Accuracy") 
plt.title('Resnet18 Knowledge Distillation with VGG16(Train)')
plt.legend() 
plt.savefig("./figures/resnet_vgg/Accuracy_train.jpg")
plt.clf()



for csv_file in os.listdir('./data_record/resnet18_vgg16'):
    df = pd.read_csv('./data_record/resnet18_vgg16/' + csv_file)
    column = df.iloc[0]
    train_losses = column.values.tolist()[1:]
    plt.plot(np.arange(len(train_losses)), train_losses,label=csv_file[0:-4])
    

plt.xlabel('epoch')
plt.ylabel("Loss") 
plt.title('Resnet18 Knowledge Distillation with VGG16(Train)')
plt.legend() 
plt.savefig("./figures/resnet_vgg/Loss_train.jpg")
plt.clf()