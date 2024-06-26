import os
import json as jsonmod
import torch

from dataset.basedataset import BaseDataset
from config import CIRR_IMAGE_DIR, CIRR_ANNOTATION_DIR

class CIRRDataset(BaseDataset):
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
        # self.image_name2path = self.load_file(os.path.join(CIRR_ANNOTATION_DIR, 'images_splits', f'split.rc2.{s}.json'))
        self.image_name2path = self.load_file(os.path.join('/data/DERI-Gong/acw557/datasets/CIRR/cirr/image_splits',f'split.rc2.{s}.json'))
        self.image_id2name = list(self.image_name2path.keys())

        # if necessary, load triplet annotations
        if self.what_elements != "target":
            # print('original annotation')
            self.annotations = self.load_file(os.path.join(CIRR_ANNOTATION_DIR, 'captions', f'cap.rc2.{s}.json'))
            print('annotation file path',os.path.join(CIRR_ANNOTATION_DIR, 'captions', f'cap.rc2.{s}.json'))
            # import pdb
            # pdb.set_trace()
            # self.annotations = self.load_file('./gpt4_blip2_triplets.json')
            # self.annotations = self.load_file('./caption_only_same.json')
            # GRB
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/CIRR_test_caption/cap.rc2.test1.ref_mod_cap_gpt4turbo_clean.json') 
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/CIRR_test_caption/cap.rc2.test1_posneg.json') # try global rank with positive only

            # text only M
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/datasets/CIRR/cirr/captions/cap.rc2.test1.json')
            # text only C
            # self.annotations = self.load_file('./captions/Ablation_cirr_textonlyC.json')
            # llava_cir
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LLaVA/results_old/LLaVA_caption_ready.json')
            # instructBLIP+gpt35 merge global
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/ablationStudy/CIRR_test_caption/GPT35_instructBLIP_global_cirr_test.json')
            # instructBLIP+gpt4 merge global
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/ablationStudy/CIRR_test_caption/GPT4Turbo_instructBLIP_global_cirr_test.json')

            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/ablationStudy/CIRR_test_caption/chatgpt35_global_cirr_test.json') # 
            
            # ablation study one prompt
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/ablationStudy/CIRR_test_caption/chatgpt4_prompt_onlyPositive_local.json') # 
            # ablation study one prompt
            # self.annotations = self.load_file('/data/DERI-Gong/acw557/project/LAVIS_CVPR/LAVIS/ablationStudy/CIRR_test_caption/chatgpt4_prompt_woexample.json') # 

            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LLaVA/captions/cap.rc2.val.llava.json')
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/gpt35_prompt1_blip2_triplets.json') 
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/gpt35_prompt1_rephrease_blip2_triplets.json') 
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/instructblip2_triplets_chatgpt35_prompt1.json')           
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/upbound_target_caption.json')           
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/blip2_triplets_chatgpt613_structure_prompt_oneexample.json')           
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/chatgpt613_1example_must.json')   
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/captions/only_target.json')        
            # print('tag caption')
            # self.annotations = self.load_file('./caption_only_noun.json')
            # self.annotations = self.load_file('/import/sgg-homes/ss014/project/LAVIS/blip2_triplets.json') # this is for BLIP only, whose caption is : blip ref caption + change to + modified text 
            # self.annotations = self.load_file('./val_location_onlylrnfbf.json')
            # self.annotations = self.load_file('./reasoning_caption.json')
            # self.annotations = self.load_file('./updown.json')
 
        self.cap_setting = 'noNeed'


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

        ann = self.annotations[index]

        capts = ann['caption']
        text, real_text = self.get_transformed_captions([capts])

        path_src = self.image_name2path[ann['reference']][2:] # remove the "./" from the relative image pathname
        img_src = self.get_transformed_image(path_src)

        path_trg = self.image_name2path[ann['target_hard']][2:] # remove the "./" from the relative image pathname
        img_trg = self.get_transformed_image(path_trg)

        return img_src, text, img_trg, real_text, index


    def get_query(self, index):
        ann = self.annotations[index]

        capts = ann['caption']
        # capts = ann['positive']
        # print(capts)
        text, real_text = self.get_transformed_captions([capts])

        path_src = self.image_name2path[ann['reference']][2:] # remove the "./" from the relative image pathname
        img_src = self.get_transformed_image(path_src)
        img_src_id = self.image_id2name.index(ann['reference'])

        if self.split == "test":
            img_trg_id = [None]
        else:
            img_trg_id = [self.image_id2name.index(ann['target_hard'])]

        return img_src, text, img_src_id, img_trg_id, real_text, index


    def get_target(self, index):

        img_id = index
        path_img = self.image_name2path[self.image_id2name[index]][2:] # remove the "./" from the relative image pathname
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


