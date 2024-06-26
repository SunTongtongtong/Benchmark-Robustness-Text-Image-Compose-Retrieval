import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import json

with open('./reasoning_captions/image_split/img_path_merged_clean.json') as f:
    image_name2path = json.load(f)

with open('./results_name.json') as f:
    results_list = json.load(f)    

root = '/import/sgg-homes/ss014/project/NeurIPS2023_Robustness/image/'
for query in results_list:
    ranked_names = [image_name2path[name] for name in query['ranked_names']]    
    caption = query['caption']
    target_name = image_name2path[query['target_hard']]
    reference_names = image_name2path[query['reference']]

    plt.figure(figsize=(12, 8))

    reference = plt.imread(os.path.join(root,reference_names),0)
    target = plt.imread(os.path.join(root,target_name),0)
    rank1 = plt.imread(os.path.join(root,ranked_names[0]),0)
    rank2 = plt.imread(os.path.join(root,ranked_names[1]),0)
    rank3 = plt.imread(os.path.join(root,ranked_names[2]),0)

    

    print(caption)
    plt.subplot(5, 1, 1)
    plt.imshow(reference)
    plt.axis('off')
    plt.subplot(5, 1, 2)
    plt.imshow(target)
    plt.axis('off')
    plt.subplot(5, 1, 3)
    plt.imshow(rank1)
    plt.axis('off')
    plt.subplot(5, 1, 4)
    plt.imshow(rank2)
    plt.axis('off')
    plt.subplot(5, 1, 5)
    plt.imshow(rank3)
    plt.axis('off')

    im_name = './evaluation_result_images_background/'+query['reference']+'.png'
    plt.title(caption)
    plt.savefig(im_name)

print('Done')
