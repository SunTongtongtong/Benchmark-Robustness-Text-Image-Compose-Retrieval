import os
import json as jsonmod
import torch

from dataset.basedataset import BaseDataset
from config import CIRR_IMAGE_DIR, CIRR_ANNOTATION_DIR

import json
from typing import List
from pathlib import Path
from PIL import Image

import numpy as np

# from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, AutoProcessor


class CIRCODataset(BaseDataset):
    """
    CIRR (Composed Image Retrieval on Real-life Images), introduced in "Image
    Retrieval on Real-Life Images With Pre-Trained Vision-and-Language Models";
    Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould;
    Proceedings of the IEEE/CVF International Conference on Computer Vision
    (ICCV), 2021, pp. 2125-2134 
    """

    def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):
        BaseDataset.__init__(self, split, CIRR_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

        # NOTE: splits are refered as train/val/test but some filepaths contain resp. "train"/"dev"/"test1"
        s = "test1" if split=="test" else split

        # load the paths of the images involved in the split
        dataset_path = Path('/data/DERI-Gong/acw557/datasets/CIRCO')
        with open(dataset_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)
        self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]

        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
        #     self.annotations: List[dict] = json.load(f)
        # ours, also change later get query

        # with open('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/circo_caption/circo.test.gpt4global.json', "r") as f:   # should be our global result
        # with open('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/circo_caption/circo.test.gpt35global.json', "r") as f:  
        # with open('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/circo_caption/circo.test.gpt4_turbo_global_instructblip.json', "r") as f:  
        with open('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/circo_caption/circo.test.gpt35global_instructblip.json', "r") as f:   
        #  
        # with open('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/circo_caption/eccv_circo.test.posneg_updateprompt.json', "r") as f:   
            self.annotations = json.load(f)
        
        print('======> annotation file loaded <======')
        # with open('/data/DERI-Gong/acw557/project/LLaVA/results_old/LLaVA_circo_pseudo_caption.json','r') as f:
        #     self.annotations = json.load(f)  

        self.max_num_gts = 23
        
        # img id, max over 123403, id in coco, index max is 123403


        self.image_name2path = {str(img_id):path for img_id, path in zip(self.img_ids,self.img_paths)} # idx in 123403 to path 
        self.image_id2name = list(self.image_name2path.keys())
        # import pdb
        # pdb.set_trace()
        # if necessary, load triplet annotations
        self.cap_setting = 'noNeed'

        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        # clip_modela, preprocess_a = clip.load("ViT-L/14", device='cuda', jit=False)  # tmlr want to change to vit14



    def __len__(self):
        if self.what_elements=='target':
            return len(self.image_id2name)
        return len(self.annotations)


    def load_file(self, f):
        """
        If the input file is the "caption" annotation file, this returns a list
        of dictionaries with the following format:
            {"pairid": 12063, 
            "reference":   "test1-147-1-img1", 
            "target_hard": "test1-83-0-img1", 
            "target_soft": {"test1-83-0-img1": 1.0}, 
            "caption": "remove all but one dog and add a woman hugging it", 
            "img_set": {"id": 1, 
                        "members": ["test1-147-1-img1", 
                                    "test1-1001-2-img0",  
                                    "test1-83-1-img1",           
                                    "test1-359-0-img1",  
                                    "test1-906-0-img1", 
                                    "test1-83-0-img1"],
                        "reference_rank": 3, 
                        "target_rank": 4}
            }
        If the input file is about the images involved in the split, this 
        returns a list of image relative path
        """
        with open(f, "r") as jsonfile:
            ann = jsonmod.loads(jsonfile.read())
        return ann


    ############################################################################
    # *** GET ITEM METHODS
    ############################################################################

    def get_triplet(self, index):
        print('--- get triplet ---')
        ann = self.annotations[index]

        capts = ann['caption']
        text, real_text = self.get_transformed_captions([capts])

        path_src = self.image_name2path[str(ann['reference_img_id'])] # remove the "./" from the relative image pathname
        img_src = self.get_transformed_image(path_src)

        path_trg = self.image_name2path[str(ann['target_img_id'])] # remove the "./" from the relative image pathname
        img_trg = self.get_transformed_image(path_trg)

        return img_src, text, img_trg, real_text, index


    def get_query(self, index):
        ann = self.annotations[index]

        # capts = ann['relative_caption']  # text M only 
        # capts = ann['reference_caption']+' change to '+ann['relative_caption']  # text C only
        # LLaVA change to ann['LLava_caption']
        # capts = ann['LLava_caption']  # text LLaVA 
        capts = ann['target_caption'] # our methods
        # print(capts)
        # capts = ann['positive'] # ablation study: prove local useful

        text, real_text = self.get_transformed_captions([capts])

        # path_src = self.image_name2path[ann['reference_img_id']] # remove the "./" from the relative image pathname
        
        reference_img_id = ann['reference_img_id']
        path_src = self.img_paths[self.img_ids_indexes_map[str(reference_img_id)]]
        img_src = self.get_transformed_image(path_src)
        img_src_id = self.img_ids_indexes_map[str(reference_img_id)]
        # self.image_id2name.index(ann['reference_img_id'])

        if self.split == "test":
            img_trg_id = [None]
        else:
            # img_trg_id = [self.image_id2name.index(ann['target_hard'])]
            # img_trg_id =[self.img_ids_indexes_map[str(ann['target_img_id'])]]
            img_trg_id =[self.img_ids_indexes_map[str(gt_id)] for gt_id in ann['gt_img_ids']]
        
        return img_src, text, img_src_id, img_trg_id, real_text, index


    def get_target(self, index):

        img_id = index
        path_img = self.image_name2path[self.image_id2name[index]] # remove the "./" from the relative image pathname
        img = self.get_transformed_image(path_img)

        return img, img_id, index


    def get_subset(self, index):
        """
        Get the ids of the images in the subset from wich the annotated
        reference image and the target image are originated. These ids are
        further used to compute the metrics R_subset@K defined for the CIRR
        dataset cf. https://github.com/Cuberick-Orion/CIRR.
        NOTE: the id of the reference image is not included in the subset !
        """ 

        ann = self.annotations[index]

        imgs_subset_ids = torch.tensor([self.image_id2name.index(im) \
                                            for im in ann['img_set']['members']
                                            if im != ann['reference']])

        return imgs_subset_ids, index


    def get_soft_targets(self, index):
        """
        Get the ids of the soft-target images for the query indexed by
        `index`, along with their qualification (1.0 when it's the actual
        target image, 0.5 when no differences with the actual target image are
        worth mentionning, -1.0 when the image is too different from the actual
        target image). Currently, this is very specific to the CIRR dataset, and
        is used at evaluation time to compute the recalls.
        cf. https://github.com/Cuberick-Orion/CIRR.
        """

        ann = self.annotations[index]

        if self.split == "test":
            softtrg_imgids_and_qualif = None
        else:
            softtrg_imgids_and_qualif = {self.image_id2name.index(im): qualif
                                        for im, qualif in ann['target_soft'].items()}

        return softtrg_imgids_and_qualif, index


    def get_pair_ids(self):
        """
        Returns each annotation's pair_id.
        Necessary to produce prediction files, to be evaluated on the server.
        """
        return [ann["pairid"] for ann in self.annotations]


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
        ann = self.annotations[index]
        return ann["caption"], ann["reference"], ann["target_hard"]
    
    def get_all_texts(self):
        """
        Returns a list of all the texts in the dataset.
        """
        return  [sublist['caption'] for sublist in self.annotations]
    
    # for robustness and cvpr; 
    def get_transformed_image(self, path):  ## Notice this is for BLIP only, other method use the above one 
        # print('############################## from raw image ##############################')
        # print(self.corruption)
       
        image = Image.open(path).convert('RGB')
        # image = image.resize((596,437))  # use for blip only 

        # transform the image (normalization & resizing + data augmentation)
        # shitong: add corrupt image here
        if self.corruption:
            try:
                image = np.array(image)
            
                # import pdb
                # pdb.set_trace()
                image = self.img_corrupt_func([image], scale=self.img_corrupt_level)[0]
                image = Image.fromarray(image) 
            except:
                print('##############################corrupt error##############################')
                print(path)
                image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
                if not self.corruption_name == 'zoom_blur_filter':
                    raise ValueError('corruption not supported')
        image = self.transform(image)
        return image

    # generate image features   
    # def get_transformed_image(self, path):  ## Notice this is for BLIP only, other method use the above one 
    #     # print('############################## from raw image ##############################')
    #     # print(self.corruption)
       
    #     image = Image.open(path).convert('RGB')

    #     inputs = self.transform(images=image, return_tensors="pt")
    #     image_features = self.clip_model.get_image_features(**inputs)
    #     return image_features




