import random
from PIL import Image

class DataSiamese():
    def __init__(self,dataDir,transform=None):
        self.dataDir=dataDir
        self.transform=transform

    def __getitem__(self,index):
        currClass=random.randint(0,1)
        data0=self.dataDir.imgs[index]
        if currClass:
            while 1:
                data1=random.choice(self.dataDir.imgs)
                if data0[1]==data1[1] and data0[0]!=data1[0]: # same class and diff imgs
                    break
        else:
            while 1:
                data1=random.choice(self.dataDir.imgs)
                if data0[1]!=data1[1]: # diff class and diff imgs
                    break
        img0=Image.open(data0[0])
        img1=Image.open(data1[0])
        if self.transform is not None:
            img0=self.transform(img0)
            img1=self.transform(img1)
        labelSiamese=currClass
        return img0,img1,labelSiamese
        
    def __len__(self):
        return len(self.dataDir.imgs)

class DataTriplet():
    def __init__(self,dataDir,transform=None):
        self.dataDir=dataDir
        self.transform=transform

    def __get_item__(self,index):
        currClass=random.randint(0,1)
        dataBase=self.dataDir.imgs[index]
        while 1:
            dataNear=random.choice(self.dataDir.imgs)
            if dataBase[1]==dataNear[1] and dataBase[0]!=dataNear[0]: # sample near
                break
        while 1:
            dataFar=random.choice(self.dataDir.imgs)
            if dataFar[1]!=dataBase[1] and dataFar[1]!=dataNear[1]: # sample far
                break

        imgBase=Image.open(dataBase[0])
        imgNear=Image.open(dataNear[0])
        imgFar=Image.open(dataFar[0])
        if self.transform is not None:
            imgBase=self.transform(imgBase)
            imgNear=self.transform(imgNear)
            imgFar=self.transform(imgFar)
        return imgBase,imgNear,imgFar
























































