#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import itertools
import json as jsonmod

from dataset.basedataset import BaseDataset
from config import FASHIONIQ_IMAGE_DIR, FASHIONIQ_ANNOTATION_DIR

class FashionIQDataset(BaseDataset):
    """
    FashionIQ dataset, introduced in "Fashion IQ: A new dataset towards
    retrieving images by natural language feedback"; Hui Wu, Yupeng Gao,
    Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio
    Feris; Proceedings of CVPR, pp. 11307–11317, 2021.
    """

    def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0,
                    fashion_categories='all', ** kw):
        """
        Args:
            fashion_categories: fashion_categories to consider. Expected to be a string such as : "dress toptee".
        """
        BaseDataset.__init__(self, split, FASHIONIQ_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)
        
        # self.gpt4_flag=True
        self.gpt4_flag=True

        self.fashion_categories = ['dress', 'shirt', 'toptee'] if fashion_categories=='all' else sorted(fashion_categories.split())

        # concatenate in one list the image identifiers of the fashion categories to consider
        image_id2name_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'image_splits', f'split.{fc}.{split}.json') for fc in self.fashion_categories]
        image_id2name = [self.load_file(a) for a in image_id2name_files]
        self.image_id2name = list(itertools.chain.from_iterable(image_id2name))

        # if necessary, load triplet annotations of the fashion categories to consider
        if self.what_elements in ["query", "triplet"]:
            # prefix = 'pair2cap' if split=='test' else 'cap'
            prefix = 'cap' 
            
            if self.gpt4_flag: # one caption 
                annotations_files = [f'/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/fashionIQcaption/ref_mod_cap_gpt4turbo_{fc}.json' for fc in self.fashion_categories]
                # annotations_files = [f'/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/fashionIQcaption/Ablation_global_psedo_gpt35_ref_mod_cap_{fc}.json' for fc in self.fashion_categories]
                # for text only C
                # annotations_files = ['/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/captions/Ablation_fashionIQ_textonlyC_{}.json'.format(fc) for fc in self.fashion_categories]
                # for llava
                # annotations_files = [f'/data/DERI-Gong/acw557/project/LLaVA/results_old/LLaVA_fashionIQ_caption_{fc}.json' for fc in self.fashion_categories]

            else: # captions 0 and 1
                annotations_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'captions', f'{prefix}.{fc}.{split}.json') for fc in self.fashion_categories]
                # for llava caption
                # annotations_files = [f'/data/DERI-Gong/acw557/project/LLaVA/results/LLaVA_fashionIQ_caption_{fc}.json' for fc in self.fashion_categories]
            annotations = [self.load_file(a) for a in annotations_files]
            self.annotations = list(itertools.chain.from_iterable(annotations))
        # decide how to deal with two captions in fashionIQ 
        # import pdb
        # pdb.set_trace()
        if kw['model_name'] in ['TIRG','ARTEMIS']:
            self.cap_setting = '<and>' # default setting as Cosmo
        elif kw['model_name'] in ['CLIP4CIR','IMAGEONLY','TEXTONLY','BLIPv2','RerankFile','Pic2word','SEARLE','BIBLIP4CIR','SPRC']: # blipv2 does not use this 
            self.cap_setting = 'and' 
        elif kw['model_name'] in ['MAAF']:
            self.cap_setting = 'inadditionto'
        else:
            raise ValueError(f'Unknown model name: {kw["model_name"]}')
        # self.cap_setting = 'inadditionto' # for MAAF: cat two captions with " inadditionto "
        # self.cap_setting = 'and'
    

    def __len__(self):
        if self.what_elements=='target':
            return len(self.image_id2name)
        
        # if self.cap_setting == '<and>':5#
            # return 2*len(self.annotations)
        else:#including self.cap_setting == 'inadditionto' and 'and'
            return 2*len(self.annotations)
        # return 2*len(self.annotations) # 1 annotation = 2 captions = 2 queries/triplets


    def load_file(self, f):
        """
        Depending on the file, returns:
            - a list of dictionaries with the following format:
                {'target': 'B001AS562I', 'candidate': 'B0088WRQVS', 'captions': ['i taank top', 'has spaghetti straps']}
            - a list of image identifiers
        """
        with open(f, "r") as jsonfile:
            ann = jsonmod.loads(jsonfile.read())
        return ann


    ############################################################################
    # *** GET ITEM METHODS
    ############################################################################

    def get_triplet(self, idx):

        # NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
        # of each reference-target pair separately, doubling the number of
        # queries
        index = idx // 2 # get the annotation index
        cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)

        # get data
        ann = self.annotations[index]

        if self.gpt4_flag:
            capts = ann['target_caption']
            real_text = capts
            text=capts
        else:
            capts = ann['captions'][cap_slice]
            text, real_text = self.get_transformed_captions(capts)

        path_src = ann['candidate'] + ".png"
        img_src = self.get_transformed_image(path_src)

        path_trg = ann['target'] + ".png"
        img_trg = self.get_transformed_image(path_trg)

        return img_src, text, img_trg, real_text, idx


    def get_query(self, idx):

        # NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
        # of each reference-target pair separately, doubling the number of
        # queries

        # if self.cap_setting == 'double':            
        index = idx // 2 # get the annotation index
        cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)
        # get data
        ann = self.annotations[index]
        # print(index)
        # print(cap_slice)

        if self.gpt4_flag:
            capts = ann['target_caption']
            text, _ = self.get_transformed_captions(capts)
            real_text = capts
            # text=capts
        else:
            capts = ann['captions'][cap_slice]
            text, real_text = self.get_transformed_captions(capts)

        # print('target captions',capts)

        path_src = ann['candidate'] + ".png"
        img_src = self.get_transformed_image(path_src)
        img_src_id = self.image_id2name.index(ann['candidate'])
        
        # print(ann)
        img_trg_id = [self.image_id2name.index(ann['target'])]

        return img_src, text, img_src_id, img_trg_id, real_text, idx


    def get_target(self, index):

        img_id = index
        path_img = self.image_id2name[index] + ".png"
        img = self.get_transformed_image(path_img)

        return img, img_id, index


    ############################################################################
    # *** FORMATTING INFORMATION FOR VISUALIZATION PURPOSES
    ############################################################################

    def get_triplet_info(self, idx):
        """
        Should return 3 strings:
            - the text modifier
            - an identification code (name, relative path...) for the reference image
            - an identification code (name, relative path...) for the target image
        """
        index = idx // 2
        cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)
        ann = self.annotations[index]
        return " [and] ".join(ann["captions"][cap_slice]), ann["candidate"], ann["target"]

    # For MAAF 
    def get_all_texts(self):
        """
        Returns a list of all the texts in the dataset.
        """
        if len(self.fashion_categories)!=3:
            raise ValueError('This is not the text for all three categories followed by MAAF')
        return  [' inadditiontothat ']+[a for sublist in self.annotations for a in sublist['captions']]