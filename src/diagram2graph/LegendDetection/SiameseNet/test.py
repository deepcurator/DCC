from __future__ import print_function
import torch
from torchvision import datasets,transforms
from torch.autograd import Variable
import argparse,os
import models
from PIL import Image
from data import DataSiamese  

def loadTestData(dataDir,dataTransform):
    
    imgData=datasets.ImageFolder(dataDir)
    testData=[]
    print(imgData)

    count=0
    while True:
        print(imgData.imgs[count][0])
        print(imgData.imgs[count+1][0])
        currImg0=Image.open(imgData.imgs[count][0])
        currImg1=Image.open(imgData.imgs[count+1][0])
        if dataTransform is not None:
            currImg0=dataTransform(currImg0)
            currImg1=dataTransform(currImg1)
            currImg0=currImg0.unsqueeze(0)
            currImg1=currImg1.unsqueeze(0)
        testData.append((currImg0,currImg1))
        count=count+2
        if count==len(imgData.imgs):
            break
    return testData
        
    
def inference(model,testData,opt):
    predictions=[]

########################################################################################
#    trainTransforms=[transforms.Resize((256,128),interpolation=3),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean,std)]
#    valTransforms=[transforms.Resize((256,128),interpolation=3),
#                   transforms.ToTensor(),
#                   transforms.Normalize(mean,std)]
#    dataTransforms={}
#    dataTransforms["train"]=transforms.Compose(trainTransforms)
#    dataTransforms["val"]=transforms.Compose(valTransforms)
#
#    imgData={}
#    imgData["train"]=DataSiamese(datasets.ImageFolder(os.path.join("/data/Market1501_toy","train")),dataTransforms["train"])
#    imgData["val"]=DataSiamese(datasets.ImageFolder(os.path.join("/data/Market1501_toy","val")),dataTransforms["val"])
#    
#    dataLoader={x:torch.utils.data.DataLoader(imgData[x],batch_size=50,shuffle=True) for x in ["train","val"]}
#    dataSizes={x:len(imgData[x]) for x in ["train","val"]}
#    currEpochCorLabel = 0
#    for data in dataLoader["train"]:
#        inp0,inp1,label=data
#        if torch.cuda.is_available():
#            inp0,inp1=Variable(inp0.cuda()),Variable(inp1.cuda())
#            label=label.cuda()
#        else:
#            inp0,inp1=Variable(inp0),Variable(inp1)
#        f0=model(inp0)
#        f1=model(inp1)        
#        _,outpBCE=model.bClassifier(f0-f1)
#        print(outpBCE)
#        print(label)
#        _, preds = torch.max(outpBCE.data, 1) #Aditi
#        print(preds)
#        corLabel = float(torch.sum(preds == label.data)) #Aditi
#        currEpochCorLabel += corLabel
#        #accurate_labels = torch.sum(torch.argmax(outpBCE, dim=1) == label).cpu()
#        #accuracy = 100. * accurate_labels / len(label.cpu().numpy())
#        #predictions.append(outpBCE)
################################################################################################################################################        
    for (img0,img1) in testData:
        if torch.cuda.is_available():
            img0,img1=Variable(img0.cuda()),Variable(img1.cuda())
        else:
            img0,img1=Variable(img0),Variable(img1)
        f0=model(img0)
        f1=model(img1)        
        _,outpBCE=model.bClassifier(f0-f1)
        _, preds = torch.max(outpBCE.data, 1) #Aditi
        print(outpBCE)
        print(preds)
        #print(outpBCE[:,0])
        predictions.append(outpBCE)
#    totAcc = 100.*(currEpochCorLabel/ dataSizes["train"]) #Aditi
#    print(totAcc)
    return predictions

if __name__=="__main__":

    
    
    parser=argparse.ArgumentParser(description="Testing")
    parser.add_argument("--gpu_ids", default="0", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2")
    #parser.add_argument("--data_dir", default="/data/Market1501_toy_test", type=str, help="testing data path")
    parser.add_argument("--data_dir", default="/data/AVA_toy_test", type=str, help="testing data path")
    parser.add_argument("--modelLoadPath", default="/data/trainedModels/re-id/trainedModelAVAToy0.pth", type=str, help="trained model save path")
    parser.add_argument("--detectionOutput",default="/data/outputs/M3output",type=str,help="Path to Module 3's output, a folder, one json per input frame")

    opt=parser.parse_args()
    dataDir=opt.data_dir
    gpuIds=[int(x) for x in opt.gpu_ids.split(',')]

    if len(gpuIds)>0:
        torch.cuda.set_device(gpuIds[0])
        
    # model
    model=models.ResNetIDE()
    model.load_state_dict(torch.load(opt.modelLoadPath))
    if torch.cuda.is_available():
        model=model.cuda()
    model=model.eval()    

    # data transforms and data loader
    mean,std=[0.485,0.456,0.406],[0.229,0.224,0.225]
    dataTransform=transforms.Compose([transforms.Resize((256,128),interpolation=3),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)])
    testData=loadTestData(dataDir,dataTransform)
    
    # run inference on test data
    predictions=inference(model,testData,opt)
    print(predictions)
                   
    
    
    
                                              
                  
    

    
    
    
    
    











































    

 



























    

 

