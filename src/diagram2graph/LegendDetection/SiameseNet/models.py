import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.init as weight_init
from torchvision import models
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import cv2,sys

def weightsInitKaiming(m):
    classname=m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weightsInitClassifier(m):
    classname=m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
        
class Classifier(nn.Module):
    def __init__(self,inputDim,nClass,dropout=True,relu=True,nBottleneck=512):
        super(Classifier, self).__init__()
        addBlock=[]
        addBlock+=[nn.Linear(inputDim,nBottleneck)]
        addBlock+= [nn.BatchNorm1d(nBottleneck,affine=True)]#[nn.BatchNorm1d(nBottleneck,affine=True,track_running_stats=True)]#
        if relu:
            addBlock+=[nn.LeakyReLU(0.1)]
        if dropout:
            addBlock+=[nn.Dropout(p=0.5)]
        addBlock=nn.Sequential(*addBlock)
        addBlock.apply(weightsInitKaiming)

        classifier=[]
        classifier+=[nn.Linear(nBottleneck,nClass)]
        
        classifier=nn.Sequential(*classifier)
        classifier.apply(weightsInitClassifier)

        self.addBlock=addBlock
        self.classifier=classifier

    def forward(self, x):
        f=self.addBlock(x)
        y=self.classifier(f)
        return f,y
        
# Define the ResNet50-based Model
class ResNetIDE(nn.Module):
    def __init__(self, droprate=0.5,stride=2):
        super(ResNetIDE, self).__init__()
        model_ft=models.resnet50(pretrained=True)
        model_ft=nn.Sequential(*list(model_ft.children())[:-2])
        if stride==1:
            model_ft.layer4[0].downsample[0].stride=(1,1)
            model_ft.layer4[0].conv2.stride=(1,1)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.bClassifier= Classifier(2048, 2, droprate)

    def forward(self, x):
        x=self.model(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),x.size(1))
        #x=self.classifier(x)
        return x		

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

 

