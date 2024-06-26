# from torchvision import models
from PIL import Image
import os.path as osp
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from torchvision import datasets,transforms,models
import os

import scipy.io as sio


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class resnet18_sst(nn.Module):
    def __init__(self,pretrain=True):
        super(resnet18_sst, self).__init__()
        self.model = models.resnet18(pretrained=pretrain)
    
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

class resnet152_sst(nn.Module):
    def __init__(self,pretrain=True):
        super(resnet152_sst, self).__init__()
        self.model = models.resnet152(pretrained=pretrain)
        # self.model = models.resnet50(pretrained=pretrain)

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        return x

class googleNet_sst(nn.Module):
    def __init__(self,pretrain=True):
        super(googleNet_sst, self).__init__()
        self.model = models.googlenet(pretrained=pretrain)
    
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.maxpool1(x)
        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool2(x)
        x = self.model.inception3a(x)
        x = self.model.inception3b(x)
        x = self.model.maxpool3(x)
        x = self.model.inception4a(x)
        x = self.model.inception4b(x)
        x = self.model.inception4c(x)
        x = self.model.inception4d(x)
        x = self.model.inception4e(x)
        x = self.model.maxpool4(x)
        x = self.model.inception5a(x)
        x = self.model.inception5b(x)
        x = self.model.avgpool(x)
        return x

class mobileNet_sst(nn.Module):
    def __init__(self,pretrain = True):
        super(mobileNet_sst,self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=pretrain)


    def forward(self,x):
        x = self.model.features(x)   
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return x

class rawDataset(Dataset):
    def __init__(self,image_paths):    
        ## this should be the transform for CIRPLANT paper 
        self.transform = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224), # or 224 
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])         # value from imagenet                  
                               ])
        self.image_path = image_paths
        # self.label = label
        
    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        img_path = self.image_path[index]
        img = read_image(img_path)
        img  = self.transform(img)
        name = img_path.split('/')[-1].split('.')[0]
        return img,name
    
def resnet152_feature(image_paths):
    model = resnet152_sst(pretrain=True).cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    dataset = rawDataset(image_paths)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1, shuffle=False)
    flag=1

    #compare with CIRPLANT pretrained features
    # import pickle
    # _img_feat = pickle.load(open('/import/sgg-homes/ss014/datasets/CIRR/cirr/img_feat_res152/dev/dev-1003-2-img0.pkl','rb')) # torch.Size([2048])  dev-1000-1-img0.pkl


    for imgs,name in tqdm(dataloader):
        imgs = imgs.cuda()
        # if flag==1:
        # if name[0] =='dev-1003-2-img0':
        #     import pdb
        #     pdb.set_trace()
        # print(name)
        features = model(imgs).squeeze().cpu().detach().numpy()
        # np.savez('/import/sgg-homes/ss014/datasets/fashionIQ/images_resnet152/'+name[0],feature = np.array(features))
            # flag=0
        # else:
            # features = np.concatenate([features,model(imgs).squeeze().cpu().detach().numpy()],0)
    # features = torch.stack(features)
    # with open('resnet18_'+opt.dataset+'.npy','wb') as f:    
        # np.save(f,np.array(features))
    # np.savez('ResNet152.npz',feature = np.array(features),label = np.array(labels))
    return features
    
if __name__ =='__main__':
    # root = '/import/sgg-homes/ss014/datasets/fashionIQ'
    # image_paths = os.listdir(root + '/' + 'images')
    # image_paths = [ os.path.join(root,'images',path) for path in image_paths]

    image_paths = '/import/sgg-homes/ss014/datasets/NLVR2/images/dev'
    image_paths = os.listdir(image_paths)
    image_paths = [ os.path.join('/import/sgg-homes/ss014/datasets/NLVR2/images/dev',path) for path in image_paths]

    resnet152_feature(image_paths)



    
    







