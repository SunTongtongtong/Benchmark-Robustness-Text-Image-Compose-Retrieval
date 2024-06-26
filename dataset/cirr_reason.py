import os
import json as jsonmod
import torch

from dataset.basedataset import BaseDataset
from config import CIRR_D_IMAGE_DIR

class CIRRDataset_reason(BaseDataset):
    """
    CIRR (Composed Image Retrieval on Real-life Images), introduced in "Image
    Retrieval on Real-Life Images With Pre-Trained Vision-and-Language Models";
    Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould;
    Proceedings of the IEEE/CVF International Conference on Computer Vision
    (ICCV), 2021, pp. 2125-2134 
    """

    def __init__(self, reasoning_type, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):
        BaseDataset.__init__(self, split, CIRR_D_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

        # NOTE: splits are refered as train/val/test but some filepaths contain resp. "train"/"dev"/"test1"
        s = "test1" if split=="test" else split


        self.image_name2path = self.load_file('./captions/reasoning_captions/image_split/img_path_merged_clean.json')

        self.image_id2name = list(self.image_name2path.keys())

        # import pdb
        # pdb.set_trace()
        # if necessary, load triplet annotations
        if self.what_elements != "target":
            # self.annotations = self.load_file(os.path.join(CIRR_ANNOTATION_DIR, 'captions', f'cap.rc2.{s}.json'))
            # self.annotations = self.load_file('./val_location_onlylrnfbf.json')
            # self.annotations = self.load_file('./numerical_reasoning_caption.json')
            # self.annotations = self.load_file('./color_reasoning_caption.json')
            # self.annotations = self.load_file('./remove_reasoning_caption.json')

            # self.annotations = self.load_file(reasoning_caption_file)
            if reasoning_type == 'remove':
                self.annotations = self.load_file('./captions/reasoning_captions/all_remove_captions.json')
            elif reasoning_type == 'attribute':
                self.annotations = self.load_file('./captions/reasoning_captions/all_attribute_captions.json')
            elif reasoning_type == 'background':
                self.annotations = self.load_file('./captions/reasoning_captions/all_background_captions.json')
            elif reasoning_type == 'numerical':
                self.annotations = self.load_file('./captions/reasoning_captions/all_numerical_captions.json')
            else:
                raise ValueError("reasoning_type should be remove, attribute, background or numerical when in CIRR_reason dataset")
           
            # self.annotations = self.load_file('./val_adj.json')

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
            softtrg_imgids_and_qualif = self.image_id2name.index(ann['target_hard'])

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


