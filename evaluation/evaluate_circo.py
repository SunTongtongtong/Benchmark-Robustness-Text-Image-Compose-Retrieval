import torch
import numpy as np
from tqdm import tqdm
# import dataset as data

import json
import os

def index2server(ext_results,args):
    # with open('/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/results/tmlr_circo_index_circo_rebuttal_time_0214.json','r') as f:
    #     ext_results = json.load(f)
    with open('/data/DERI-Gong/acw557/datasets/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', "r") as f:
        imgs_info = json.load(f)

    img_ids = [img_info["id"] for img_info in imgs_info["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}


    result = ext_results['0']
    # result = [{idx:result[:50]} for idx,result in enumerate(ext_results)]

    result_clear = {}
    for idx,line in enumerate(result):
        result_clear[str(idx)] = [img_ids[index] for index in line[:50]]


    file_path = './results/{}_{}_circo_server_test.json'.format(args.model,args.exp_name)
    with open(file_path,'w') as f:
        json.dump(result_clear,f)
    print('CIRCO submission file saved in {}'.format(file_path))

def index2server_fromfile(file):
    with open(file,'r') as f:
        ext_results = json.load(f)
    with open('/data/DERI-Gong/acw557/datasets/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', "r") as f:
        imgs_info = json.load(f)
    img_ids = [img_info["id"] for img_info in imgs_info["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}
    result = ext_results['0']
    # result = [{idx:result[:50]} for idx,result in enumerate(ext_results)]

    result_clear = {}
    import pdb
    pdb.set_trace()
    for idx,line in enumerate(result):
        result_clear[str(idx)] = [img_ids[index] for index in line[:50]]

    name = file.split('/')[-1]
    txt_dit = '/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_text/results/circo_server_txt'
    file_path = os.path.join(txt_dit,'{}_circo_server_test.json'.format(name))
    with open(file_path,'w') as f:
        json.dump(result_clear,f)
    print('CIRCO submission file saved in {}'.format(file_path))




def deal_with_CIRCO(args, vocab, cs_sorted_ind, split):
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    results[0] : shape [220, 100] 
    """

    # Generate the predicted features
    # predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, relative_val_dataset,
                                                                                 #    ref_names_list, pseudo_tokens)
    _, targets_loader = data.get_eval_loaders(args, vocab, split)
    dataset = targets_loader.dataset
    annotations = dataset.annotations


    # gts_img_ids = []    
    # for idx, ann in enumerate(annotations):
    #     img_trg_id =[dataset.img_ids_indexes_map[str(gt_id)] for gt_id in ann['gt_img_ids']]
    #     gts_img_ids.append(img_trg_id)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    for sorded_ind, ann in tqdm(zip(cs_sorted_ind, annotations)):
        ann['gt_img_ids'] = [str(num) for num in ann['gt_img_ids']]
        gt_img_ids =  np.array(ann['gt_img_ids'])
        index_names = [str(name) for name in np.array(dataset.img_ids)]
        sorted_index_names = np.array(index_names)[sorded_ind]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        # assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == gt_img_ids[0])
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100
       
    # Move the features to the device
    # index_features = index_features.to(device)
    # predicted_features = predicted_features.to(device)

    # Normalize the features
    # index_features = F.normalize(index_features.float())

    # for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
    #     gt_img_ids = np.array(gt_img_ids)[
    #         np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
    #     similarity = predicted_feature @ index_features.T
    #     sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    #     sorted_index_names = np.array(index_names)[sorted_indices]
    #     map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
    #     precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
    #     precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

    #     ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
    #     ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
    #     ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
    #     ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

    #     assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
    #     single_gt_labels = torch.tensor(sorted_index_names == target_name)
    #     recall_at5.append(float(torch.sum(single_gt_labels[:5])))
    #     recall_at10.append(float(torch.sum(single_gt_labels[:10])))
    #     recall_at25.append(float(torch.sum(single_gt_labels[:25])))
    #     recall_at50.append(float(torch.sum(single_gt_labels[:50])))



    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


# for generate testing files TMLR
# if __name__ == "__main__":

#     # root = '/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/results/tmlr_circo'
#     # files = os.listdir(root)
#     # files.remove('circo_server')
#     # files.remove('corruption_index_result')
    
#     root='/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_text/results'
#     files = os.listdir(root)

#     idx=0
#     for file in files:
#         print(idx)
#         index2server_fromfile(os.path.join(root,file))
#         idx+=1
#     # index2server(None,None)
  
  # for save image features 
#   if __name__ =="__main__":
#     image_feature_generate()