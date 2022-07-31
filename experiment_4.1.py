import numpy
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas
import torchvision
from torchvision import transforms
import torch.nn.functional as F

"""
The network is referenced from https://arxiv.org/pdf/1503.02531v1.pdf
"""




learning_rate=0.0001
torch.manual_seed(0)

##### change this part to use different temperature
temp = 7
drop_out=0.3
alpha = 0.5
batchSize=32
#####



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
train_dataset = torchvision.datasets.MNIST(root="mnist_dataset/",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root="mnist_dataset/",train=False,transform=transforms.ToTensor(),download=True)



class TeacherModel(nn.Module):
    def __init__(self, ):
        super(TeacherModel, self).__init__()
        self.full_connect1 = nn.Linear(784, 1200)
        self.full_connect2 = nn.Linear(1200, 1200)
        self.full_connect3 = nn.Linear(1200, 10)
        self.dt = nn.Dropout(drop_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.full_connect1(x)
        x = self.dt(x)
        x = self.relu(x)
        x = self.full_connect2(x)
        x = self.dt(x)
        x = self.relu(x)
        x = self.full_connect3(x)
        return x  


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.full_connect1 = nn.Linear(784,15)
        self.full_connect2 = nn.Linear(15, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.full_connect1(x)
        x = self.relu(x)
        x = self.full_connect2(x)
        return x  

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

def getAcc(target_model):
    corr = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            preds = target_model(x)
            predictions = preds.max(1).indices
            corr += (predictions == y).sum()
            total += predictions.size(0)
        acc = (torch.true_divide(corr,total)).item()
        print(acc)
        target_model.train()
    return target_model

def execute(epochs,target_model):
    for epoch in range(0,epochs):
        target_model.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            preds = target_model(data)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        target_model.eval()
        target_model=getAcc(target_model)
        
        
    return target_model.eval()





teacher_model = TeacherModel()
teacher_model = teacher_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate)
print("Teacher model:")
teacher_model=execute(6,teacher_model)
torch.save(teacher_model.state_dict(),"./teacher_model.pkl")
student_model = StudentModel()
student_model = student_model.to(device)
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
print("Student model:")
student_model=execute(6,student_model)

torch.save(student_model.state_dict(),"./student_model.pkl")
student_model = StudentModel()
student_model = student_model.to(device)
student_model.train()
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
soft_loss = nn.KLDivLoss(reduction="batchmean")
hard_loss = nn.CrossEntropyLoss()



print("ditillation:")

print("train with temp "+ str(temp))
for epoch in range(0,6):
    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            teacher_preds = teacher_model(data)
        student_preds = student_model(data)
        student_loss = hard_loss(student_preds, targets)
        ditillation_loss = soft_loss(
            F.softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    student_model.eval()
    student_model=getAcc(student_model)

src="./student_model_distillation"+"_temp"+str(temp)+".pkl"
torch.save(student_model.state_dict(),src)
    