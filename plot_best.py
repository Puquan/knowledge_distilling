import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


best_result_vgg = pd.read_csv('./data_record/resnet18_vgg16/Resnet18-vgg16_T=5,beta=0.5.csv')
best_result_resnet = pd.read_csv('./data_record/resnet18_resnet50/Resnet18-ResNet50_T=10,beta=0.5.csv')


column1 = best_result_vgg.iloc[3]
column2 = best_result_resnet.iloc[3]
train_losses1 = column1.values.tolist()[1:]
train_losses2 = column2.values.tolist()[1:]
plt.plot(np.arange(len(train_losses1)), train_losses1,label='Resnet18-vgg16_T=5,beta=0.5')
plt.plot(np.arange(len(train_losses2)), train_losses2,label='Resnet18-ResNet50_T=10,beta=0.5')
plt.xlabel('epoch')
plt.ylabel("Accuracy") 
plt.title('Comparison of Distillation between Resnet50 and VGG16(Test)')
plt.legend() 
plt.savefig("./figures/best/Accuracy_test.jpg")
plt.clf()



column1 = best_result_vgg.iloc[2]
column2 = best_result_resnet.iloc[2]
train_losses1 = column1.values.tolist()[1:]
train_losses2 = column2.values.tolist()[1:]
plt.plot(np.arange(len(train_losses1)), train_losses1,label='Resnet18-vgg16_T=5,beta=0.5')
plt.plot(np.arange(len(train_losses2)), train_losses2,label='Resnet18-ResNet50_T=10,beta=0.5')
plt.xlabel('epoch')
plt.ylabel("Loss") 
plt.title('Comparison of Distillation between Resnet50 and VGG16(Test)')
plt.legend() 
plt.savefig("./figures/best/Loss_test.jpg")
plt.clf()

column1 = best_result_vgg.iloc[1]
column2 = best_result_resnet.iloc[1]
train_losses1 = column1.values.tolist()[1:]
train_losses2 = column2.values.tolist()[1:]
plt.plot(np.arange(len(train_losses1)), train_losses1,label='Resnet18-vgg16_T=5,beta=0.5')
plt.plot(np.arange(len(train_losses2)), train_losses2,label='Resnet18-ResNet50_T=10,beta=0.5')
plt.xlabel('epoch')
plt.ylabel("Accuracy") 
plt.title('Comparison of Distillation between Resnet50 and VGG16(Train)')
plt.legend() 
plt.savefig("./figures/best/Accuracy_train.jpg")
plt.clf()

column1 = best_result_vgg.iloc[0]
column2 = best_result_resnet.iloc[0]
train_losses1 = column1.values.tolist()[1:]
train_losses2 = column2.values.tolist()[1:]
plt.plot(np.arange(len(train_losses1)), train_losses1,label='Resnet18-vgg16_T=5,beta=0.5')
plt.plot(np.arange(len(train_losses2)), train_losses2,label='Resnet18-ResNet50_T=10,beta=0.5')
plt.xlabel('epoch')
plt.ylabel("Loss") 
plt.title('Comparison of Distillation between Resnet50 and VGG16(Train)')
plt.legend() 
plt.savefig("./figures/best/Loss_train.jpg")
plt.clf()