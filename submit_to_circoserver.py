import json

# self.img_paths = [dataset_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
#                           imgs_info["images"]]

with open('/data/DERI-Gong/acw557/datasets/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', "r") as f:
    imgs_info = json.load(f)

img_ids = [img_info["id"] for img_info in imgs_info["images"]]

img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

        # with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
        #     self.annotations: List[dict] = json.load(f)
 
# with open('./results/circo/GRB/neurips_circo_index_circo_GRB_openclip_TEXTONLY.json','r') as f:     
# with open('./results/circo/TextonlyM/neurips_circo_index_circo_neurips_openclip_bigG_textonly_TEXTONLY.json','r') as f:
# with open('./results/circo/TextonlyC/neurips_circo_index_circo_TextonlyC_openclip_TEXTONLY.json','r') as f:
with open('./results/neurips_circo_index_circo_gpt4_instructblip_openclip_TEXTONLY.json','r') as f:

    result = json.load(f)
    result = result['0']

result_clear = {}
for idx,line in enumerate(result):

    result_clear[str(idx)] = [img_ids[index] for index in line[:50]]
    # result_clear[str(idx)] = line[:50]

with open('./results/circo_server_test.json','w') as f:
    json.dump(result_clear,f)
# print(result.keys())


####### after global
# with open('/data/DERI-Gong/acw557/project/LLaVA/ranking/rerank_result/TEXTONLY_rerank.json', 'r') as f:
#     result = json.load(f)

# result_clear = {}
# for idx,line in enumerate(result):
#     result_clear[str(idx)] = [img_ids[index] for index in line[str(idx)][:50]]
#     # result_clear[str(idx)] = line[:50]

# with open('./results/circo_server_test.json','w') as f:
#     json.dump(result_clear,f)
