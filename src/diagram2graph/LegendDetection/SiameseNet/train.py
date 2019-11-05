from __future__ import print_function
#%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets,transforms
import argparse,os
import models
from data import DataSiamese


def loadModels(network,modelPath):
    pretrainedState=torch.load(modelPath)
     
    pretrainedState.pop("model.fc.weight")
    pretrainedState.pop("model.fc.bias")
    
    tmp=[str(k) for k,_ in pretrainedState.items() if "classifier" in k]
    for x in tmp:
        pretrainedState.pop(x)
        
    temp_list = list(pretrainedState.items())
    
    for k,v in temp_list:
        k1=k
        if "layer" in k:
            #print("Here layer")
            k=k.replace("layer","")
            pretrainedState[k]=v
            pretrainedState.pop(k1)
        if "model.conv1" in k:
            #print("Here model.conv1")
            k=k.replace("model.conv1","model.0")
            pretrainedState[k]=v
            pretrainedState.pop(k1)
        if "model.bn1" in k:
            #print("Here bn1")
            k=k.replace("model.bn1","model.1")
            pretrainedState[k]=v
            pretrainedState.pop(k1)         
            
    count=0
    temp_list = list(pretrainedState.items())

    for k,v in temp_list:
        k1=k
        if count>=6:
            currChar=k[6]
            k=k.replace("model."+currChar,"model."+str(int(currChar)+3))
            pretrainedState[k]=v
            pretrainedState.pop(k1)        
        count=count+1

    networkState=network.state_dict()
    networkState.update(pretrainedState)
    network.load_state_dict(networkState)
    return network
    
def trainModel(model,dataLoader,dataSizes,criterion,optimizer,opt):
    
    totLoss={}
    totLoss["train"]=[]
    totLoss["val"]=[]
    
    totAccuracy={}
    totAccuracy["train"]=[]
    totAccuracy["val"]=[]
    
    print(len(dataLoader["train"].dataset))
    
    train_embeddings = []
    train_labels = []
    val_embeddings = []
    val_labels = []
        
    for epoch in range(opt.nEpochs):
        scheduler.step(epoch)
        print("Epoch {}/{}".format(epoch,opt.nEpochs))
        
        for phase in ["train","val"]:
            if phase=="train":
                model.train(True)
            else:
                model.train(False)
                
            currEpochLoss = 0.0  
            currEpochCorLabel = 0.0 #Aditi
            for data in dataLoader[phase]:
                inp0,inp1,label=data
                if torch.cuda.is_available():
                    inp0,inp1=Variable(inp0.cuda()),Variable(inp1.cuda())
                    label=label.cuda()
                else:
                    inp0,inp1=Variable(inp0),Variable(inp1)
                    
                optimizer.zero_grad()
                
                f0=model(inp0)
                f1=model(inp1)
                
#                print(f0.shape)
                
                temp,outpBCE=model.bClassifier(f0-f1)
                
                _, preds = torch.max(outpBCE.data, 1) #Aditi
                
                if epoch == opt.nEpochs-1:
                    
                    if phase=="train":
                        train_embeddings.append(temp.data.cpu().numpy())
                        train_labels.append(label.cpu().numpy())
                        
                    if phase=="val":
                        val_embeddings.append(temp.data.cpu().numpy())
                        val_labels.append(label.cpu().numpy())
                                                
                
                loss = 0.5*criterion(outpBCE,label)
                corLabel = float(torch.sum(preds == label.data)) #Aditi
                #print(loss)
                
                if phase=="train":
                    loss.backward()
                    optimizer.step()
                currEpochLoss+=loss.item()*len(label.cpu().numpy()) #Aditi
                #currEpochLoss+=loss.item()
                currEpochCorLabel += corLabel #Aditi
                
                if phase == "val":
                    accurate_labels = torch.sum(torch.argmax(outpBCE, dim=1) == label).cpu()
                    accuracy = 100. * accurate_labels / len(label.cpu().numpy())
                     
                    print('No. of accurate label =%d, No. of total Label = %d, Test accuracy: %f, Loss: %f, currEpochLoss = %f, corLabel = %f, currEpochCorLabel = %f'%(accurate_labels, len(label.cpu().numpy()), accuracy, loss, currEpochLoss, corLabel, currEpochCorLabel))
   
         
            #print('----------------------currEpochLoss = {}, dataSizes[{}] = {} -------------------------------------'.format(currEpochLoss, phase, dataSizes[phase]))
       
            epochLoss = currEpochLoss/dataSizes[phase]
            totLoss[phase].append(epochLoss)
            epochAcc = 100.*(currEpochCorLabel/ dataSizes[phase]) #Aditi
            totAccuracy[phase].append(epochAcc) #Aditi
            print("--------------------------phase {} bce loss: {} Acc: {}------------------------------".format(phase,epochLoss, epochAcc))
   
    
    fig, ax = plt.subplots()
    ax.plot(totLoss["train"]) 
    ax.plot(totLoss["val"]) 
    ax.plot(totAccuracy["train"]) 
    ax.plot(totAccuracy["val"]) 
    plt.show()
    
    torch.save(model.cpu().state_dict(),opt.modelSavePath)     
    return train_embeddings, train_labels, val_embeddings, val_labels
    
    
if __name__=="__main__":    

    parser=argparse.ArgumentParser(description="Training")
    parser.add_argument("--gpu_ids", default="0", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2")
    #parser.add_argument("--data_dir", default="/data/Market1501_small", type=str, help="training data path")
    parser.add_argument("--data_dir", default="/data/AVA_toy", type=str, help="training data path")
    parser.add_argument("--batch_size", default=12, type=int, help="training batch size")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate") #0.03
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight decay") #5e-4
    parser.add_argument("--nEpochs", default=150, type=int, help="number of epochs")
    parser.add_argument("--pretrainedPath", default="/data/trainedModels/re-id/resnet50_IDE.pth", type=str, help="pretrained ide model")
    parser.add_argument("--modelSavePath", default="/data/trainedModels/re-id/trainedModelAvaToy1.pth", type=str, help="trained model save path")

    opt=parser.parse_args()

    dataDir=opt.data_dir
    gpuIds=[int(x) for x in opt.gpu_ids.split(',')]

    if len(gpuIds)>0:
        torch.cuda.set_device(gpuIds[0])
        
    # model
    model=models.ResNetIDE()
    #print(model)
    model=loadModels(model,opt.pretrainedPath)
    if torch.cuda.is_available():
        model=model.cuda()
    

    # data transforms and data loader
    mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
    trainTransforms=[transforms.Resize((256,128),interpolation=3),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean,std)]
    valTransforms=[transforms.Resize((256,128),interpolation=3),
                   transforms.ToTensor(),
                   transforms.Normalize(mean,std)]
    dataTransforms={}
    dataTransforms["train"]=transforms.Compose(trainTransforms)
    dataTransforms["val"]=transforms.Compose(valTransforms)

    imgData={}
    imgData["train"]=DataSiamese(datasets.ImageFolder(os.path.join(dataDir,"train")),dataTransforms["train"])
    imgData["val"]=DataSiamese(datasets.ImageFolder(os.path.join(dataDir,"val")),dataTransforms["val"])
    
    dataLoader={x:torch.utils.data.DataLoader(imgData[x],batch_size=opt.batch_size,shuffle=True) for x in ["train","val"]}
    dataSizes={x:len(imgData[x]) for x in ["train","val"]}
    
    # optimizer, criterion, lr scheduler etc
    optimizer = optim.SGD(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay,momentum=opt.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1) #Aditi
   
    criterion=nn.CrossEntropyLoss()
    
    # train model
    train_embeddings, train_labels, val_embeddings, val_labels = trainModel(model,dataLoader,dataSizes,criterion,optimizer,opt)
    

    
    
    
    
    
    
                                              
                  
    

    
    
    
    
    











































    

 

