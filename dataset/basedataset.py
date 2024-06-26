#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import sys
import pickle

import nltk
# nltk.download('punkt') # NOTE (should be done once, if not installed when setting the environment)

import torch
import torch.utils.data as data

from PIL import Image
# NOTE: tackle error "OSError: image file is truncated (7 bytes not processed)"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import cleanCaption

import corrupt.image_corrupt as img_crpt
# import corrupt.text_corrupt as txt_crpt
import numpy as np
from utils.resnet152_extract import resnet152_sst
from torchvision import datasets,transforms,models


class BaseDataset(data.Dataset):
    """
    Umbrella class for datasets for image search with (free-form) text modifiers.
    """

    def __init__(self, split, img_dir, vocab, transform, what_elements='triplet', load_image_feature=0,**kw):
        """
        Args:
            - split: train|val|test, to get the right data
            - img_dir: root directory where to look for the dataset images
            - vocab: vocabulary wrapper, to encode the words
            - transform: function to transform the images (data augmentation,
              crop, normalization ...)
            - what_elements: element(s) to provide when when iterating over the
              dataset (calling __getitem__) (triplet, querie, target...)
            - load_image_feature: whether to load raw images (if 0, default) or
              pretrained image feature (if > 0, size of the feature)
        """
        self.split = split
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.what_elements = what_elements
        self.find_get_item_func()

        # shitong: add corrupt image here
        self.corruption = False
        self.corruption_text = False
        if 'img_corrupt' in kw and 'img_corrupt_level' in kw:
            if kw['img_corrupt']!= 'none':
                self.corruption = True
                self.corruption_name = kw['img_corrupt']
                self.img_corrupt_func = getattr(img_crpt,kw['img_corrupt'])
                self.img_corrupt_level = kw['img_corrupt_level']
        # if 'txt_corrupt' in kw:
        #     if kw['txt_corrupt']!= 'none':
        #         self.corruption_text = True
        #         self.corruption_name_text = kw['txt_corrupt']
        #         self.txt_corrupt_func = getattr(txt_crpt,kw['txt_corrupt'])
                
        if load_image_feature:
            self.size_of_loaded_feature = load_image_feature
            self.get_transformed_image = self.load_image_feature
            self.resnet152 = resnet152_sst(pretrain=True)
            for param in self.resnet152.parameters():
                param.requires_grad = False
            self.resnet152.eval()
            self.transform = transforms.Compose([transforms.Resize(224),
                transforms.CenterCrop(224), # or 224 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])         # value from imagenet                  
                ])
    def find_get_item_func(self):
        # shitong : maybe add corrupt here later 
        if self.what_elements=="triplet":
            self.get_item_func = self.get_triplet
        elif self.what_elements=="query":
            self.get_item_func = self.get_query
        elif self.what_elements=="target":
            self.get_item_func = self.get_target
        # --- additionally, for CIRR:
        elif self.what_elements=='subset':
            self.get_item_func = self.get_subset
        elif self.what_elements == "soft_targets":
            self.get_item_func = self.get_soft_targets
        else:
            print("Dataloader: unknown use case! (asked for '{}')".format(self.what_elements))
            sys.exit(-1)

    def __getitem__(self, index):
        return self.get_item_func(index)

    ############################################################################
    # *** FROM DATA TO TENSORS (imgages or text)
    ############################################################################

    # # default function
    def get_transformed_image(self, path):
        # print('############################## from raw image ##############################')
        # print(self.corruption)
        if 'replace' in path:
            print(path)
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        # transform the image (normalization & resizing + data augmentation)
        # shitong: add corrupt image here
        if self.corruption:
            # print(self.img_corrupt_level)            
            try:
                image = np.array(image)
                # print(image.shape)
                # print(path)
                image = self.img_corrupt_func([image], scale=self.img_corrupt_level)[0]
                image = Image.fromarray(image) 
            except:
                print('##############################corrupt error##############################')
                print(path)
                image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
                if not self.corruption_name == 'zoom_blur_filter':
                    raise ValueError('corruption not supported')
        # print('helloo',image.size)
        image = self.transform(image)
        # print('hello',image.size())
        return image
    
    # def get_transformed_image(self, path):  ## Notice this is for BLIP only, other method use the above one CVPR paper use this 
    #     print('############################## from raw image ##############################')
    #     print(self.corruption)
    #     if 'replace' in path:
    #         print(path)
    #     image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
    #     print('hello image size',image.size)
    #     image = image.resize((596,437))  # use for blip only 

    #     # transform the image (normalization & resizing + data augmentation)
    #     # shitong: add corrupt image here
    #     if self.corruption:
    #         try:
    #             image = np.array(image)
    #             # print(image.shape)
    #             # print(path)
    #             image = self.img_corrupt_func([image], scale=self.img_corrupt_level)[0]
    #             image = Image.fromarray(image) 
    #         except:
    #             print('##############################corrupt error##############################')
    #             print(path)
    #             image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
    #             if not self.corruption_name == 'zoom_blur_filter':
    #                 raise ValueError('corruption not supported')
    #     image = self.transform(image)
    #     return image

    # shitong: add corrupt image here
    # def get_corrpted_image(self,path):
    # 	image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
    # 	image = motion_blur_filter([image], scale=1)[0]
    # 	image = Image.fromarray(image)
    # 	# transform the image (normalization & resizing + data augmentation)
    # 	image = self.transform(image)
    # 	return image


    # def load_image_feature(self, path): # for quick test, change to the func below later
    #     # load feature directly from file
    #     path = os.path.join(self.img_dir, path).replace(".png", ".pkl").replace(".jpg", ".pkl")
    #     try:
    #         image = torch.tensor(pickle.load(open(path, "rb"))) # shape eg. (self.size_of_loaded_feature)
    #     except FileNotFoundError:
    #         print("File not found: {}".format(path))
    #         return torch.zeros(self.size_of_loaded_feature)
    #     return image

    # shitong rewrite, to load raw image and add corrupt, but use freeze resnet152
    def load_image_feature(self, path):
        print('############################## from freeze resnet image ##############################')
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        print(path)
        # transform the image (normalization & resizing + data augmentation)
        #change pkl to freeze resnet152
        if self.corruption:
            print(self.img_corrupt_func)
            # print(self.img_corrupt_level)
            image = np.array(image)
            try:
                image = self.img_corrupt_func([image], scale=self.img_corrupt_level)[0]
            except:
                print('##############################corrupt error##############################')
                print('error occure when transfer to zoom ',path )
               
            image = Image.fromarray(image) 
        image = self.transform(image)
        image = self.resnet152(image.unsqueeze(0)).squeeze()
        return image


    def get_transformed_captions(self, capts, vocab=True):
        """
        Convert sentences (string) to word ids.
        """
        # print('=======')
        # print(capts)
        if self.corruption_text:           
            
            capts = self.txt_corrupt_func(capts[0])[0]  # discard the distance here
            # print(capts)
            # print('=======')
        tokens_capts = [[] for i in range(len(capts))]
        for i in range(len(capts)):
            tokens_capts[i] = nltk.tokenize.word_tokenize(cleanCaption(capts[i]).lower())

        if self.cap_setting =='<and>':
            ret_capts = " <and> ".join(capts)
        elif self.cap_setting == 'and':
            ret_capts = " and ".join(capts)
        else:
            ret_capts = " inadditionto ".join(capts)
        if len(capts) == 1:
            tokens = tokens_capts[0]
        else:
            if self.cap_setting == '<and>':
                tokens = tokens_capts[0] + ['<and>'] + tokens_capts[1]
            elif self.cap_setting == 'and':
                tokens = tokens_capts[0] + [' and '] + tokens_capts[1]
            else:
                tokens = tokens_capts[0] + [' inadditionto '] + tokens_capts[1]

        if vocab:
            sentence = []
            sentence.append(self.vocab('<start>'))
            sentence.extend([self.vocab(token) for token in tokens])
            sentence.append(self.vocab('<end>'))
            text = torch.Tensor(sentence)

        return text, ret_capts


    ############################################################################
    # *** GET ITEM METHODS
    ############################################################################

    def get_triplet(self, index):
        raise NotImplementedError

    def get_query(self, index):
        raise NotImplementedError

    def get_target(self, index):
        raise NotImplementedError

    # --- additionally, for CIRR:

    def get_subset(self, index):
        raise NotImplementedError

    def get_soft_targets(self, index):
        raise NotImplementedError


    ############################################################################
    # *** FORMATTING INFORMATION FOR VISUALIZATION PURPOSES
    ############################################################################

    def get_triplet_info(self, index):
        """
        Should return 3 strings:
            - the text modifier
            - an identification code (name, relative path...) for the reference image
            - an identification code (name, relative path...) for the target image
        """
        raise NotImplementedError