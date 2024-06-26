import torch
import numpy as np
from tqdm import tqdm
import json
# for reference
def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics

def deal_with_COCO(args, vocab, cs_sorted_ind, split):
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    results[0] : shape [220, 100] 
    """
    ground_truth = torch.arange(len(cs_sorted_ind)).view(-1, 1)
    # for ranking in cs_sorted_ind:
    metrics = {}
    preds = torch.where(cs_sorted_ind == ground_truth)[1]
    preds = preds.detach().cpu().numpy()

    # metrics[f"{name}_mean_rank"] = preds.mean() + 1
    # metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = np.mean(preds < k)
    for key, value in metrics.items():
        print(f"{key}: {value}")

       
    # Move the features to the device
    # index_features = index_features.to(device)
    # predicted_features = predicted_features.to(device)

from PIL import Image
import os.path as osp
import pandas as pd
import os
def visualization(index):
    with open('/data/DERI-Gong/acw557/project/LLaVA/ranking/rerank_result/TEXTONLY_rerank.json','r') as f:
        results = json.load(f)
    # with open('/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/data/coco/coco_img_path.json','r') as f:
    #     image_name2path = json.load(f)    
    # image_id2name = list(image_name2path.keys())
    with open('/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/data/coco/coco_triplets.json','r') as f:
        annotations = json.load(f)

    COCO_IMAGE_DIR = '/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/data/coco/val2017'
    csv_file = osp.join(COCO_IMAGE_DIR, 'coco_eval.csv')
    df = pd.read_csv(csv_file, sep=",")                
    img_paths = df['id'].tolist()
    img_target_paths = [osp.join(COCO_IMAGE_DIR,'val2017',name) for name in img_paths]              
    img_query_paths = [osp.join(COCO_IMAGE_DIR,'val2017_masked_crop',name) for name in img_paths]

    k=10

    img_path = [img_query_paths[index]]    
    topk_index = results[index][str(index)][:k]
    tar_img_files = [img_target_paths[index] for index in topk_index]
    img_path.extend(tar_img_files)
    images = [Image.open(file) for file in img_path]
    widths, heights = zip(*(image.size for image in images))
    # 确定新图片的尺寸
    new_width = sum(widths)
    new_height = max(heights)

    # 创建新图片对象
    new_image = Image.new('RGB', (new_width, new_height))

    # 将图片粘贴到新图片中
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width
    # 保存拼接后的图片
    print(annotations[index]['caption'])
    new_image.save(os.path.join(f'./index_{index}.jpg'))     
    # print('saving image grid to', os.path.join(image_root_folder, prompt,'COCO_retrievaled_concatenated_image.jpg'))       

if __name__ == '__main__':
    # deal_with_COCO(args, vocab, cs_sorted_ind, split)
    visualization(19)
    print('Done!   ') 