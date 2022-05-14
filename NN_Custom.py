#First step just classification
#Creating the model
#Custome NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
#Architettura 8 layer

# due da 32 conv
# quattro da 64 conv

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #Abbiamo kernel 5x5
        #Provare a diminuire i filtri
        self.conv1 = nn.Conv2d(3, 32, 5)# da 32
        self.conv2 = nn.Conv2d(32, 64, 5)# da 64
        self.conv3 = nn.Conv2d(64, 64, 5)# da 64
        self.conv4 = nn.Conv2d(64, 128, 5)# da 128
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(881600,512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512,128)  # 5*5 from image dimension
        self.out = nn.Linear(128,3) #Mettere anche uno

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Net()
print(model)
print(torch.cuda.is_available())
import torch
from torchvision import transforms, datasets
data_transform = transforms.Compose([
        transforms.RandomResizedCrop((227,227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = optim.SGD(model.parameters(), 0.1)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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