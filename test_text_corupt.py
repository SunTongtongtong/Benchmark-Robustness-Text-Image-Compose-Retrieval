from corrupt.text_corrupt import *
import parrot

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=3)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

# num_beams = 20
# num_return_sequences = 20
# text = "an animal runs through an icy tunnel in a video game"
# ret = get_response(text,num_return_sequences,num_beams)

# print(text)
# print(ret)

from corrupt.text_corrupt import *

# model_name = 'tuner007/pegasus_paraphrase'

        
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# para_model = PegasusForConditionalGeneration.from_pretrained(model_name).to('cpu')        
        
# sent_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device='cpu')

# # caption = "an animal runs through an icy tunnel in a video game"
# caption = 'Add another three bird of the same kind.'

# scales = 3

# paraphrases, bert_scores = paraphrase_filter(caption, scales, para_model, tokenizer, sent_model)

# print(paraphrases)
# print(bert_scores)


# print('############')
# # text = 'a boy watching the news while holding his teddy bear'
# text = ["Shows a dog of the same breed resting its head on someone's knee."]
# from corrupt.text_corrupt import *
# result = character_filter(text[0], 3, 'jaro')
# corrupted_sent, levenshtein_dist = result

# print(corrupted_sent)
# print(levenshtein_dist)

# print('###########')


# from Levenshtein import distance
# distance(text, 'ah anmal rns through a icy unnel i a video game')

# text = 'a boy watching the news while holding his teddy bear'
# text = 'Add another three bird of the same kind.'
# from corrupt.text_corrupt import *
import corrupt.text_corrupt as txt_crpt

TEXT_CORRUPTS = ['character_filter','qwerty_filter','RemoveChar_filter','remove_space_filter',  'misspelling_filter', 'repetition_filter','homophones_filter']
# text_sample = 'Just two big birds with five little ones.'
# text_sample = 'There were two adult dogs on the road - there was one grown puppy in the yard.'
# text_sample = 'has black and white pattern.'
text_sample = 'Put the parrot in the basket with toys'
for corrupt in TEXT_CORRUPTS:
# for i in range(20):
  # corrupt='remove_space_filter'
  text_corrupt_func = getattr(txt_crpt,corrupt)
  corrupted_sent, levenshtein_dist = text_corrupt_func(text_sample, 3)
  print(corrupt)
  print(corrupted_sent)
  print(levenshtein_dist)