# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models
from torchvision import transforms, datasets


#Transfer learning

data_transform = transforms.Compose([
        transforms.RandomResizedCrop((227,227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#Specifico per VGG
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trainset = datasets.ImageFolder(root='Concrete Crack Images for Classification/train',
                                           transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=32, shuffle=True,
                                             num_workers=2)
testset = datasets.ImageFolder(root='Concrete Crack Images for Classification/valid',
                                           transform=data_transform)
testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=32, shuffle=False,
                                             num_workers=2)

# loading the pretrained model
model = models.vgg16_bn(pretrained=True)
resnet18 = models.resnet152(pretrained=True)


# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()

# Add on classifier
model.classifier[6] = Sequential(Linear(4096, 2))
for param in model.classifier[6].parameters():
    param.requires_grad = True

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = Adam(model.classifier[6].parameters(), lr=0.0005)


NUM_EPOCHS = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(NUM_EPOCHS):  

    running_loss = 0.0
    print(epoch)
    for i, (inputs, labels) in enumerate(trainloader):
        
        # 0) get the inputs
        inputs, labels = inputs.to(device), labels.to(device)     
        # 1) zero the gradients
        optimizer.zero_grad()               
        # 2) forward
        outputs = model(inputs)
        # 3) compute loss
        loss = criterion(outputs, labels)   
        # 4) backward
        loss.backward()                     
        # 5) optimization step
        optimizer.step()                    
        
        # optionally: print statistics
        running_loss += loss.item()
        if i % 3 == 2:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch+1, i+1, running_loss/3))
            running_loss = 0.0


torch.save(model.state_dict(), "model_prod")

listAcc = []
for i in range(0,50):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)     
            outputs = model(inputs)
            #print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            print("Labels : ")
            print(labels)
            print("Predicted : ")
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
listAcc.append((100 * correct / total))
sum=0
for i in listAcc:
    sum += i

print(sum/len(listAcc))

from torchsummary import summary
summary(model, input_size=(3, 600, 400))