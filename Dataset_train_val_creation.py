import os
from os import listdir
from os.path import isfile, join
import random

import shutil

#Script for creating the train and val distribution

mypath = "Concrete Crack Images for Classification/Positive"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
os.mkdir("Concrete Crack Images for Classification/valid")
validPos = "Concrete Crack Images for Classification/valid/Positive"
os.mkdir(validPos)
validNeg = "Concrete Crack Images for Classification/valid/Negative"
os.mkdir(validNeg)
print(len(onlyfiles))

listOfValidPositive = set()
while(len(listOfValidPositive) < 6000):
    random.seed()
    randInt = random.randint(0, 19999)
    listOfValidPositive.add(onlyfiles[randInt])
    print(len(listOfValidPositive))
print("Randomization Negative End")
for i in listOfValidPositive:
    os.replace(mypath+"/"+str(i),validPos+"/"+str(i))
print("End Replace")

mypath = "Concrete Crack Images for Classification/Negative"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

listOfValidNegative =set()
while(len(listOfValidNegative) < 6000):
    random.seed()
    randInt = random.randint(0, 19999)
    listOfValidNegative.add(onlyfiles[randInt])
    print(len(listOfValidNegative))
print("Randomization Negative End")
for i in listOfValidNegative:
    os.replace(mypath+"/"+str(i),validNeg+"/"+str(i))
print("End Replace")
# absolute path
trainDir = "Concrete Crack Images for Classification/train"
os.mkdir(trainDir)
mypath = "Concrete Crack Images for Classification/Positive"
shutil.move(mypath, trainDir)
mypath = "Concrete Crack Images for Classification/Negative"
shutil.move(mypath, trainDir)