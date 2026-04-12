# from corrupt.utils import get_k_biggest_index
from corrupt.corruption import *

from multiprocessing import Pool
import tqdm

import json
import os
import random
import math

import torch
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from sentence_transformers import SentenceTransformer

from wildnlp.aspects.utils import compose
from wildnlp.aspects import RemoveChar, Swap, QWERTY, Misspelling, Articles

import Levenshtein

def get_k_biggest_index(scores, k):
    if type(k) == int:
        k = [k]
    # return the index of k-th largest scores
    return list(np.argsort(-scores)[k])



def insert_char_filter(texts, scale):
    # randomly inserts characters or spaces
    pass

def swap_char_filter(texts, scale):
    # randomly swaps characters or spaces
    pass

def keyboard_filter(texts, scale):
    # Simulates errors made while writing on a QWERTY-type keyboard.
    pass




def synonym_filter(texts, scale):
    # randomly replaces words with synonyms
    pass


def word_dropout_filter(texts, scale):
    # randomly removes words
    pass

def stop_words_filter(texts, scale):
    # randomly removes stop words
    pass

def misagreement_filter(texts, scale):
    # randomly replaces words with words that have opposite meaning
    pass

def plural_filter(texts, scale):
    # randomly replaces words with plural form
    pass


def missing_article_filter(texts, scale):
        c = [.08, .12, 0.18, 0.26, 0.38][scale - 1]


def repetition(texts, seed=42):
    # randomly repeats words
    words = texts.split(' ')
    rep_pos = random.choice(range(len(words)))
    # insert an element in a list
    words.insert(rep_pos, words[rep_pos])
    return ' '.join(words)



def character_filter(texts, num_scale=0, method='levenshtein'):
    ch_count = len(texts)
    word_count = len(texts.split(' '))
    # corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.01, 0.04, 0.09]]
    corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.04]]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):
            
            crpt_texts = Swap(transform_percentage=1.5/word_count, seed=random.randint(1,100))(crpt_texts)               
            

        ret.append(crpt_texts)

    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]
    return ret, dist


def qwerty_filter(texts,  num_scale=0, method='levenshtein'):
    ch_count = len(texts)
    word_count = len(texts.split(' '))
    corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.04]]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):           
            
            crpt_texts = QWERTY(words_percentage=1, characters_percentage=1, seed=random.randint(1,100))(crpt_texts)         
            

        ret.append(crpt_texts)

    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist

def RemoveChar_filter(texts,  num_scale=0, method='levenshtein'):
    ch_count = len(texts)
    word_count = len(texts.split(' '))
    corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.04]]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):
            crpt_texts = RemoveChar(words_percentage=1, characters_percentage=1, seed=random.randint(1,100))(crpt_texts)           

        ret.append(crpt_texts)

    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist

def remove_space_filter(texts,  num_scale=0, method='levenshtein'):
    ch_count = len(texts)
    word_count = len(texts.split(' '))
    corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.04]]
    # corrupt_count = [math.ceil(ch_count * percentage) for percentage in [0.10]]


    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):
            crpt_texts = RemoveChar(' ', words_percentage=1, characters_percentage=1, seed=random.randint(1,100))(crpt_texts)           

        ret.append(crpt_texts)

    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist


def misspelling_filter(texts,  num_scale=0, method='levenshtein'):
    # corrupt_count = [1,2,3]
    corrupt_count = [2]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):            
            crpt_texts = Misspelling(use_homophones=False, seed=random.randint(1,100))(crpt_texts)
            if Levenshtein.distance(texts, crpt_texts) == 0:
                crpt_texts = repetition(crpt_texts, seed=random.randint(1,100))          

        ret.append(crpt_texts)


    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist


def repetition_filter(texts,  num_scale=0, method='levenshtein'):
    # corrupt_count = [1,2,3]
    corrupt_count = [2]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):
            crpt_texts = repetition(crpt_texts, seed=random.randint(1,100))          

        ret.append(crpt_texts)


    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist


def homophones_filter(texts,  num_scale=0, method='levenshtein'):
    # corrupt_count = [1,2,3]
    corrupt_count = [2]

    ret = []

    for crpt in corrupt_count:
        crpt_texts = texts
        for i in range(crpt):           
            crpt_texts = Misspelling(use_homophones=True, seed=random.randint(1,100))(crpt_texts)
            if Levenshtein.distance(texts, crpt_texts) == 0:
                crpt_texts = Misspelling(use_homophones=False, seed=random.randint(1,100))(crpt_texts)

        ret.append(crpt_texts)
    dist_func = {'levenshtein': Levenshtein.distance, 'jaro': Levenshtein.jaro, 'jaro_winkler': Levenshtein.jaro_winkler}[method]
    dist = [dist_func(texts, x) for x in ret]

    return ret, dist


def misagreement_filter(texts,  num_scale=0, method='levenshtein'):
    pass


# sentence-level
def paraphrase_filter(text, num_scale, para_model, para_tokenizer, sent_model):
    # randomly replaces sentences with similar sentences
    def get_response(input_text,num_return_sequences,num_beams):
        batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(para_model.device)
        translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=3)
        tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    
    num_beams = 20
    num_return_sequences = 20
    ret = get_response(text,num_return_sequences,num_beams)

    import numpy as np
    all_sentences = [text] + ret
    sentence_embeddings = sent_model.encode(all_sentences, normalize_embeddings=True)
    bert_scores = np.matmul(sentence_embeddings[0], sentence_embeddings[1:].T)

    k_list = [round((lv+1)/num_scale*num_return_sequences)-1 for lv in range(num_scale)]
    indices = get_k_biggest_index(bert_scores, k_list)

    return [ret[idx] for idx in indices], list(bert_scores[indices])


class ParaCorruption(Corruption):
    def __init__(self, device='cpu'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c_type = []

    @abstractmethod
    def filter(self, frames, scale, **kwargs):
        raise NotImplementedError

    def run(self, video_ids: list, original_captions: str, new_captions:str, scales:3, **kwargs):
        # remove file if existing
        if os.path.exists(new_captions):
            os.remove(new_captions)


        model_name = 'tuner007/pegasus_paraphrase'
        model_name = '.cache/pegasus'
        sentence_model = '.cache/bert-base-nli-mean-tokens'
        
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        para_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)        
        
        sent_model = SentenceTransformer(sentence_model, device=self.device)

        # open json file
        with open(original_captions) as f:
            anno = json.load(f)['annotations']

        curruped_captions = []

        with tqdm.tqdm(total=len(anno)) as pbar:
            for c in anno:
                if c['image_id'] in video_ids:
                    # get caption
                    caption = c['caption']

                    # paraphrase
                    paraphrases, bert_scores = paraphrase_filter(caption, scales, para_model, tokenizer, sent_model)
                    bert_scores = map(lambda x: str(x), bert_scores)
                    
                    curruped_captions.append({'image_id':c['image_id'], 'id':c['id'], 'currupt': list(zip(paraphrases, bert_scores))})

                pbar.update(1)

        # save to json file
        with open(new_captions, 'w') as json_file:
            json.dump({'paraphrase':curruped_captions}, json_file, 
                    indent=4,  
                    separators=(',',': '))

        print('corruption done')
                


class CharacterCorruption(Corruption):
    def __init__(self, device='cpu'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c_type = []

    @abstractmethod
    def filter(self, frames, scale, **kwargs):
        raise NotImplementedError

    def run(self, video_ids: list, original_captions: str, new_captions:str, scales:3, **kwargs):
        # remove file if existing
        if os.path.exists(new_captions):
            os.remove(new_captions)

        # open json file
        with open(original_captions) as f:
            anno = json.load(f)['annotations']

        curruped_captions = []

        with tqdm.tqdm(total=len(anno)) as pbar:
            for c in anno:
                if c['image_id'] in video_ids:
                    # get caption
                    caption = c['caption']

                    # paraphrase
                    paraphrases, scores = character_filter(caption, scales, method='jaro')
                    scores = map(lambda x: str(x), scores)
                    
                    curruped_captions.append({'image_id':c['image_id'], 'id':c['id'], 'currupt': list(zip(paraphrases, scores))})

                pbar.update(1)

        # save to json file
        with open(new_captions, 'w') as json_file:
            json.dump({'paraphrase':curruped_captions}, json_file, 
                    indent=4,  
                    separators=(',',': '))

        print('corruption done')
                


class MisspellingCorruption(Corruption):
    def __init__(self, device='cpu'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.c_type = []

    @abstractmethod
    def filter(self, frames, scale, **kwargs):
        raise NotImplementedError

    def run(self, video_ids: list, original_captions: str, new_captions:str, scales:3, **kwargs):
        # remove file if existing
        if os.path.exists(new_captions):
            os.remove(new_captions)

        # open json file
        with open(original_captions) as f:
            anno = json.load(f)['annotations']

        curruped_captions = []

        with tqdm.tqdm(total=len(anno)) as pbar:
            for c in anno:
                if c['image_id'] in video_ids:
                    # get caption
                    caption = c['caption']

                    # paraphrase
                    paraphrases, scores = misspelling_filter(caption, scales, method='jaro')
                    scores = map(lambda x: str(x), scores)
                    
                    curruped_captions.append({'image_id':c['image_id'], 'id':c['id'], 'currupt': list(zip(paraphrases, scores))})

                pbar.update(1)

        # save to json file
        with open(new_captions, 'w') as json_file:
            json.dump({'misspelling':curruped_captions}, json_file, 
                    indent=4,  
                    separators=(',',': '))

        print('corruption done')
                
