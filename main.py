from utils.operator import *
# from utils.evaluate import *
from torch.utils.data import DataLoader

import os
import time
from tqdm import tqdm
import pickle
import copy
import torch

from opts import arg_parser,verify_input_args
import dataset as data
# from models import *
from evaluation.evaluate_cirr import deal_with_CIRR
from evaluation.evaluate_circo import deal_with_CIRCO,index2server
from evaluation.evaluate_fashionIQ import deal_with_fashionIQ
from evaluation.evaluate_coco import deal_with_COCO
from evaluation.evaluate_cirr_reason import deal_with_CIRR_reason

from dataset.vocab import Vocabulary

from config import RERANK_FILE_DIR
import json
import os.path as osp
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
####
####



def validate(model, args, vocab, transform=None,output_type="metrics", max_retrieve = 100, split='val'): # shitong 
    """
    Input:
        model, args, vocab;
        output_type: either "metrics" or "rankings",
        max_retrieve: top number of propositions to keep for a given query,
        split;

    Output:
    - if output_type is "metrics": returns a message presenting the results and
        a validation score. If applicable, results are presented for each data
        category.
    - if output_type is "rankings": tensor of size (#queries, max_retrieved)
        containing the top ranked target ids corresponding to each query. If
        applicable, results are organized per data category.
    """

    # Special case for CIRR: metrics are computed at the end, based on the rankings
    # output_type_inpractice = "rankings" if (args.data_name == "cirr" or args.data_name =='cirr_reason') else output_type
    # output_type_inpractice = "rankings" if args.data_name == "cirr" else output_type
    output_type_inpractice = "rankings" if args.data_name == "cirr" or args.data_name=='circo'  or args.data_name=='fashionIQ' else output_type
    # output_type_inpractice = "rankings" if args.data_name == "cirr" or args.data_name=='circo'  else output_type

    # Initializations
    results = []
    categories = args.name_categories if ("all" in args.categories) else args.categories.split(' ') # if not applicable, `categories` becomes `[None]``

    # Switch to eval mode
    model.eval()
    # Compute measures or rankings
    if not args.model == 'RerankFile':
        output_type_inpractice = "rankings" if args.data_name == "cirr" or args.data_name=='circo' or args.data_name=='cirr_reason' or args.data_name=='coco' or args.data_name=='imgnet'  else output_type

        for category in categories:

            # specify the category to be studied, if applicable
            opt = copy.deepcopy(args)
            if args.study_per_category and (args.number_categories > 1):
                opt.categories = category

            # load data
            queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, split,transform = transform)
            # compute & process compatibility scores 
            ## shitong add to improve accurary
            max_retrieve = targets_loader.dataset.__len__()
            ####################### shitong add

            with torch.no_grad(): # no need to retain the computational graph and gradients
                start = time.time()
                res = compute_and_process_compatibility_scores(queries_loader, targets_loader,
                                                        model, opt, output_type_inpractice,
                                                        max_retrieve)
                end = time.time()
                print("\nProcessing time : ", end - start)
                
            # store results for presentation / further process
           
            results.append(res)
    else:
        # RerankFile: load the results from a file
        if args.data_name == 'cirr':
            with open(osp.join(RERANK_FILE_DIR,'TEXTONLY_rerank.json'), 'r') as f:
                results.append(json.load(f))
                args.recall_k_values = [1, 5, 10,50]
                values = [value for d in results[0] for value in d.values()]
                results = [torch.tensor(values)]
        elif args.data_name == 'coco':
            with open(osp.join('/data/DERI-Gong/acw557/project/LLaVA/ranking/rerank_result/','TEXTONLY_rerank.json'), 'r') as f:
                results.append(json.load(f))
                values = [value for d in results[0] for value in d.values()]
                results = [torch.tensor(values)]

        else: 
            with open('/data/DERI-Gong/acw557/project/LLaVA/ranking/rerank_result/FashionIQ_rerank.json', 'r') as f: # rerank result 
            # with open('/data/DERI-Gong/acw557/project/NeurIPS2023_Robustness_CVPR/NeurIPS2023_Robustness/results/rebuttal_index_debug_fashionIQ.json','r') as f:  # global retrieval fashionIQ result

            # check whether original generated index can lead to the same result, replace to generated rank index later TODO  
                results=json.load(f)

                # args.recall_k_values = [1, 5, 10,50]
                # values = [value for d in results[0] for value in d.values()]
                # results = [torch.tensor(values)]
    ## shitong save for rerank retrieval 
    if not args.model == 'RerankFile':
        if args.data_name == 'cirr':
            num_res = {idx:item.tolist() for idx,item in enumerate(results)}
            # with open("results/index_{}_test{}_gpt4turbo.json".format(args.model,args.data_name), 'w') as f:
            # with open("results/index_{}_test{}.json".format(args.model,args.data_name), 'w') as f:
            with open("results/neurips_index_{}_{}.json".format(args.exp_name,args.data_name), 'w') as f:
                json.dump(num_res, f)
        if args.data_name == 'circo':
            num_res = {idx:item.tolist() for idx,item in enumerate(results)}  # convert tensor to list 
            # with open("results/rebuttal_circo_index_{}_0130.json".format(args.exp_name), 'w') as f:
            with open("results/neurips_circo_index_{}_{}.json".format(args.exp_name,args.model), 'w') as f:
                json.dump(num_res, f)
            print('INDEX FILE saved for CIRCO')
            index2server(num_res,args)
        if args.data_name == 'fashionIQ':            
            num_res= {idx:item.tolist() for idx,item in enumerate(results)}   
     
            with open("results/neurips_index_{}.json".format(args.exp_name), 'w') as f:
                json.dump(num_res, f)
        if args.data_name == 'coco':
            num_res= {idx:item.tolist() for idx,item in enumerate(results)}            
            with open("results/rebuttal_index_{}_{}.json".format(args.exp_name,args.data_name), 'w') as f:
                json.dump(num_res, f)
        if args.data_name == 'imgnet':
            num_res= {idx:item.tolist() for idx,item in enumerate(results)}            
            with open("results/rebuttal_index_{}_{}.json".format(args.exp_name,args.data_name), 'w') as f:
                json.dump(num_res, f)
            print('saved file location: ',"results/rebuttal_index_{}_{}.json".format(args.exp_name,args.data_name))
        if args.data_name == 'cirr_reason':
            num_res= {idx:item.tolist() for idx,item in enumerate(results)}            
            with open("results/rebuttal_index_{}_{}_{}.json".format(args.exp_name,args.model,args.data_name), 'w') as f:
                json.dump(num_res, f)
            print('saved file location: ',"results/rebuttal_index_{}_{}_{}.json".format(args.exp_name,args.model,args.data_name))

    if output_type=="metrics":
        # compute additional metrics and present properly the results
        if args.data_name == "cirr" :
            # also compute the subset ranking
            message, val_mes = deal_with_CIRR(args, vocab, results[0], 	split) # [0] because no category in CIRR 
        elif args.data_name == 'circo':
            results = deal_with_CIRCO(args, vocab, results[0], split) 
            for k, v in results.items():
                print(f"{k} = {v:.2f}")
        elif args.data_name == 'coco':
            results = deal_with_COCO(args, vocab, results[0], split)
            for k, v in results.items():
                print(f"{k} = {v:.2f}")
        elif args.data_name == "cirr_reason":
            # also compute the subset ranking
            message, val_mes = deal_with_CIRR_reason(args, vocab, results[0], 	split) 
        else:
            print('==>',results.__len__())
            print('==>',results[0].__len__())

            if not args.model == 'RerankFile':
                message, val_mes = results_func(results, args)
            else:
                if args.data_name == 'fashionIQ':
                    message, val_mes = deal_with_fashionIQ(args, vocab, results, split)
                
        return message, val_mes
    if args.data_name == 'coco':
        message, val_mes = deal_with_COCO(args, vocab, results, split)

    return results


def compute_and_process_compatibility_scores(data_loader_query, data_loader_target,
                                        model, args, output_type="metrics",
                                        max_retrieve=50):
    """
    Compute the compatibility score of each query of the query dataloader with
    regard to all the candidate targets of the target dataloader, and process it.
    To save some memory at evaluation time, this function should be called "with
    torch.no_grad()".

    Input:
        output_type: either "metrics" or "rankings"

    Output:
    - if output_type is "metrics": tensor of size (#queries) containing the rank
        of the best ranked correct target for each query;
    - if output_type is "rankings": tensor of size (#queries, max_retrieved)
          containing the top ranked target ids corresponding to each query.
    """

    nb_queries= len(data_loader_query.dataset)
    results_name={}
    # Initialize output
    if output_type=="metrics":
        # return the rank of the best ranked correct target
        ret = torch.zeros(nb_queries, requires_grad=False)
    else:
        # return the top propositions for each query
        ret = torch.zeros(nb_queries, max_retrieve, requires_grad=False).int()
    ret_scores = np.zeros((nb_queries, max_retrieve)) # shitong for collecting score 

    # Pre-compute image embeddings (includes all target & reference images)

    ########### todo: use this, now for debug
    # if args.data_name in ['fashionIQ']:
    #     if os.path.exists('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,nb_queries)):
    #         with open('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,nb_queries),'rb') as f:
    #             all_img_embs = pickle.load(f)
    #     else:
    #         all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)
    #         with open('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,nb_queries),'wb') as f:
    #             pickle.dump(all_img_embs,f)
    # elif args.data_name in ['cirr_reason']:
    #     if os.path.exists('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,args.reasoning_type)):
    #         with open('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,args.reasoning_type),'rb') as f:
    #             all_img_embs = pickle.load(f)
    #     else:
    #         all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)
    #         with open('./precompute/{}_{}_{}.pkl'.format(args.data_name,args.model,args.reasoning_type),'wb') as f:
    #             pickle.dump(all_img_embs,f)
    # else:
    #     if os.path.exists('./precompute/{}_{}.pkl'.format(args.data_name,args.model)):
    #         with open('./precompute/{}_{}.pkl'.format(args.data_name,args.model),'rb') as f:
    #             all_img_embs = pickle.load(f)
    #     else:
    #         all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)
    #         with open('./precompute/{}_{}.pkl'.format(args.data_name,args.model),'wb') as f:
    #             pickle.dump(all_img_embs,f)
    ################
    all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)


    # save CLIP ViT-L/14 features for retrieval later 

    # all_img_embs = torch.zeros((123403,32,256)).cuda() # to change 
    # Compute and process compatibility scores (process by batch)
 
    if args.model in ['SPRC']:
        all_img_embs_compact = torch.cat([tensor[0].unsqueeze(0) for tensor in all_img_embs])

    for data in tqdm(data_loader_query):

        # Get query data
        image_src, txt, txt_len, img_src_ids, img_trg_ids, raw_texts, indices = data
        if torch.cuda.is_available():
            txt, txt_len = txt.cuda(), txt_len.cuda()
        # Compute query embeddings for the whole batch
        # (the reference image embedding is included in `all_img_embs`, so there
        # is only the text embedding left to compute)
        if args.model in ['ARTEMIS','TIRG']:
            txt_embs = model.get_txt_embedding(txt, txt_len) 

        elif args.model in ['MAAF']: # for model need raw text input
            txt_embs = model.get_txt_embedding(raw_texts) # tuple with both text embedding and text mask
            batch_img_embeddins = model.img_model(image_src.cuda())
            # img_src
        elif args.model in ['CLIP4CIR','BIBLIP4CIR','SPRC'] or 'OPENCLIP4CIR' in args.model:
            txt_embs = model.get_txt_embedding(raw_texts) 

        elif args.model in ['CIRPLANT']:
            txt_embs = model.get_txt_embedding(raw_texts) 
            batch_img_embeddins = model.get_source_image_embedding(image_src.cuda())
            # add image embedding 
        elif args.model in ['IMAGEONLY']:
            txt_emb = torch.zeros(1,1)
        elif args.model in ['TEXTONLY','BLIPv2','instructBLIP','BLIP4CIR','Pic2word','SEARLE']:
            txt_embs = model.get_txt_embedding(raw_texts)
        
        else:
            raise ValueError("Model not supported")
        # Process each query of the batch one by one
        for i, index in enumerate(indices):

            # Select data related to the current query
            if not args.model in ['IMAGEONLY']:
                txt_emb = txt_embs[i]
            img_src_id = img_src_ids[i]
            GT_indices = img_trg_ids[i]
            img_src_emb = all_img_embs[img_src_id]

            if args.model in ['MAAF','CIRPLANT']:
                img_src_emb = batch_img_embeddins[i]

            # Compute compatibility scores between the query and each candidate target
            if args.model in ['SPRC']:
                cs = model.get_compatibility_from_embeddings_one_query_multiple_targets(
                                        img_src_emb, txt_emb, all_img_embs_compact)
            else:    
                cs = model.get_compatibility_from_embeddings_one_query_multiple_targets(
                                        img_src_emb, txt_emb, all_img_embs)
            
            # Remove the source image from the ranking
            if args.data_name not in ['coco','imgnet']:  # for coco, the source image is not in the target list 
                cs[img_src_id] = float('-inf')

            # Rank targets
            cs_sorted_ind = cs.sort(descending=True)[1]
            
            # Store results           

            if output_type == "metrics":
                ret[index] = get_rank_of_GT(cs_sorted_ind, GT_indices)[0]
            else:
                ret[index, :max_retrieve] = cs_sorted_ind[:max_retrieve].cpu().int()
            ret_scores[index, :max_retrieve] = cs[cs_sorted_ind][:max_retrieve].cpu()
            

    ##############
    # with open('clip_textonly_results_name.json','w') as f:
    #     json.dump(results_name,f)
    ret_scores = ret_scores.tolist()
    
    with open('./results/ranking_scores_neurIPS_{}_{}_{}.json'.format(args.exp_name,args.data_name,nb_queries),'w') as f:
    # with open('./results/neurips_ranking_scores_{}_{}.json'.format(args.exp_name,args.data_name),'w') as f:
        json.dump(ret_scores,f)        
        
    ##############
    return ret


def compute_necessary_embeddings_img(data_loader_target, model, args):

    """
    Compute the embeddings of the target images.
    To save some memory, this function should be called "with torch.no_grad()".

    Input:
        data_loader_target: dataloader providing images and indices of the provided
            items within the dataloader
        model, args;

    Output:
        img_trg_embs (cuda)
    """

    img_trg_embs = None
    for data in tqdm(data_loader_target):

        # Get target data
        img_trg, _, indices = data
        indices = torch.tensor(indices)
        if torch.cuda.is_available():
            img_trg = img_trg.cuda()

        # Compute embedding
    
        img_trg_emb = model.get_image_embedding(img_trg)
        # Initialize the output embeddings if not done already
        if img_trg_embs is None:
            if args.model in ['BLIPv2']:
                # emb_sz = [len(data_loader_target.dataset), img_trg_emb.shape[1],img_trg_emb.shape[2],img_trg_emb.shape[3]]    # shitong before change blip2 tp quick version
                emb_sz = [len(data_loader_target.dataset), img_trg_emb.shape[1],img_trg_emb.shape[2]]    
                img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False).cuda()
            elif args.model in ['BLIP4CIR']:
                emb_sz = [len(data_loader_target.dataset), img_trg_emb.shape[1],img_trg_emb.shape[2]]    
                img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False).cuda()
            elif args.model in ['instructBLIP','SPRC']:
                emb_sz0 = [len(data_loader_target.dataset), img_trg_emb[0].shape[1],img_trg_emb[0].shape[2]]    
                emb_sz1 = [len(data_loader_target.dataset), img_trg_emb[1].shape[1],img_trg_emb[1].shape[2]]    
                img_trg_embs_0 = torch.zeros(emb_sz0, dtype=img_trg_emb[0].dtype, requires_grad=False).cuda()
                img_trg_embs_1 = torch.zeros(emb_sz1, dtype=img_trg_emb[1].dtype, requires_grad=False).cuda()
                img_trg_embs = 1 # dummy, broke the if condition
            # elif args.model in ['SPRC']:
            #     emb_sz = [len(data_loader_target.dataset), img_trg_emb.shape[1],img_trg_emb.shape[2]]    
            #     img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False).cuda()
            else:
                emb_sz = [len(data_loader_target.dataset), img_trg_emb.shape[1]]
                img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False).cuda()
            # if torch.cuda.is_available():
            #     img_trg_embs = img_trg_embs.cuda()

        # Preserve the embeddings by copying them
        if torch.cuda.is_available():
            if args.model in ['instructBLIP','SPRC']:
                img_trg_embs_0[indices] = img_trg_emb[0]
                img_trg_embs_1[indices] = img_trg_emb[1]
            else:
                img_trg_embs[indices] = img_trg_emb
        else :
            img_trg_embs[indices] = img_trg_emb.cpu()

    if args.model in ['instructBLIP','SPRC']:
        img_trg_embs = [(t1,t2) for t1, t2 in zip(img_trg_embs_0,img_trg_embs_1)]         # in this case, not a tensor, but a tuple of two tensors
    return img_trg_embs


def get_rank_of_GT(sorted_ind, GT_indices):
    """
    Get the rank of the best ranked correct target provided the target ranking
    (targets are identified by indices). Given two acceptable correct targets of
    respective indices x and y, if the target of index x has a better rank than
    the target of index y, then the returned value for `rank_of_GT ` is the rank
    of the target of index x, and the value of `best_GT` is x.

    Input:
        sorted_ind: tensor of size (number of candidate targets), containing the
            candidate target indices sorted in decreasing order of relevance with
            regard to a given query.
        GT_indices: list of correct target indices for a given query.

    Output:
        rank_of_GT: rank of the best ranked correct target, if it is found
            (+inf is returned otherwise)
        best_GT: index of the best ranked correct target

    """

    rank_of_GT = float('+inf')
    best_GT = None
    for GT_index in GT_indices:
        tmp = torch.nonzero(sorted_ind == GT_index)
        if tmp.size(0) > 0: # the GT_index was found in the ranking
            tmp = tmp.item()
            if tmp < rank_of_GT:
                rank_of_GT = tmp
                best_GT = GT_index
    return rank_of_GT, best_GT


def get_recall(rank_of_GT, K):
    return 100 * (rank_of_GT < K).float().mean()


def results_func(results, args):
    """
    Compute metrics over the dataset and present them properly.
    The result presentation and the computation of the metric might depend
    on particular options/arguments (use the `args`).

    Input:
        results: list containing one tensor per data category (or just one
            tensor if the dataset has no particular categories). The tensor is
            of size (number of queries) and ontains the rank of the best ranked
            correct target.
        args: argument parser from option.py

    Ouput:
        message: string message to print or to log
        val_mes: measure to monitor validation (early stopping...)
    """

    nb_categories = len(results)
    # --- Initialize a dictionary to hold the results to present
    H = {"r%d"%k:[] for k in args.recall_k_values}
    H.update({"medr":[], "meanr":[], "nb_queries":[]})

    # --- Iterate over categories
    for i in range(nb_categories):
        # get measures about the rank of the best ranked correct target
        # for category i
        for k in args.recall_k_values:
            H["r%d"%k].append(get_recall(results[i], k))
        H["medr"].append(torch.floor(torch.median(results[i])) + 1)
        H["meanr"].append(results[i].mean() + 1)
        H["nb_queries"].append(len(results[i]))

    # --- Rearrange results (aggregate category-specific results)
    H["avg_per_cat"] = [sum([H["r%d"%k][i] for k in args.recall_k_values])/len(args.recall_k_values) for i in range(nb_categories)]
    val_mes = sum(H["avg_per_cat"])/nb_categories
    H["nb_total_queries"] = sum(H["nb_queries"])
    for k in args.recall_k_values:
        H["R%d"%k] = sum([H["r%d"%k][i]*H["nb_queries"][i] for i in range(nb_categories)])/H["nb_total_queries"]
    H["rsum"] = sum([H["R%d"%k] for k in args.recall_k_values])
    H["med_rsum"] = sum(H["medr"])
    H["mean_rsum"] = sum(H["meanr"])

    # --- Present the results of H in a single string message
    message = ""

    # multiple-category case: print category-specific results
    if nb_categories > 1:
        categories = args.name_categories if ("all" in args.categories) else args.categories
        cat_detail = ", ".join(["%.2f ({})".format(cat) for cat in categories])

        message += ("\nMedian rank: " + cat_detail) % tuple(H["medr"])
        message += ("\nMean rank: " + cat_detail) % tuple(H["meanr"])
        for k in args.recall_k_values:
            message += ("\nMetric R@%d: " + cat_detail) \
                        % tuple([k]+H["r%d"%k])

        # for each category, average recall metrics over the different k values
        message += ("\nRecall average: " + cat_detail) % tuple(H["avg_per_cat"])

        # for each k value, average recall metrics over categories
        # (remove the normalization per the number of queries)
        message += "\nGlobal recall metrics: {}".format( \
                        ", ".join(["%.2f (R@%d)" % (H["R%d"%k], k) \
                        for k in args.recall_k_values]))

    # single category case
    else:
        message += "\nMedian rank: %.2f" % (H["medr"][0])
        message += "\nMean rank: %.2f" % (H["meanr"][0])
        for k in args.recall_k_values:
            message += "\nMetric R@%d: %.2f" % (k, H["r%d"%k][0])

    message += "\nValidation measure: %.2f\n" % (val_mes)

    return message, val_mes


def load_model(args):
    # Load vocabulary

    vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')

    assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
    
    vocab = pickle.load(open(vocab_path, 'rb'))
   
    print('Loading models ...')
    # Setup model
    if args.model == "TIRG":
        from models import TIRG
        model = TIRG(args,vocab.word2idx)
        model.load_state_dict(torch.load(args.ckpt)['model'])
        transform = 'default'

    elif args.model == "ARTEMIS":
        from models import ARTEMIS
        # model version is ARTEMIS or one of its ablatives
        model = ARTEMIS(args,vocab.word2idx)
        
        model.load_state_dict(torch.load(args.ckpt)['model'])
        transform = 'default'

    elif args.model == "MAAF":
        from models import MAAF
        from dataset.fashioniq import FashionIQDataset
        from dataset.cirr import CIRRDataset
        from dataset.cirr_reason import CIRRDataset_reason
        if args.data_name == 'fashionIQ':
            load_text = FashionIQDataset('train', vocab, None, 'triplet',
                    args.load_image_feature, fashion_categories='all', img_corrupt=args.img_corrupt , img_corrupt_level=args.img_corrupt_level,model_name=args.model)
        elif args.data_name == 'cirr' or args.data_name == 'cirr_reason':
            load_text = CIRRDataset('train', vocab, None, what_elements='triplet', load_image_feature=0)
         
        # elif args.data_name == 'cirr_reason':
        #     load_text = CIRRDataset_reason(args.reasoning_caption,'train', vocab, None, what_elements='triplet', load_image_feature=0)
  
        else:
            raise ValueError('Unknown dataset: {}'.format(args.data_name))

        model = MAAF(args,load_text.get_all_texts())
        #loading
        model.load_state_dict(torch.load(args.ckpt)['model_state_dict'])
        transform = 'default'

    elif args.model == 'CIRPLANT':
        import pytorch_lightning
        from models import CIRPLANT
        model = CIRPLANT(args)
        model = model.load_from_checkpoint(args.ckpt, hparams_file=None) # must manually load weights 
        model.eval()
        model=model.cuda()
        transform = 'default'

    elif args.model == 'CLIP4CIR':
        from models import CLIP4CIR
        # from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
        model= CLIP4CIR(args)
        model.eval()

        transform = model.preprocess
        # combining_function = model.combine_features
    elif args.model == 'FASHIONVIL':
        print('Loading FashionVIL model ...')
        from models import FASHIONVIL
        model = FASHIONVIL(args)  
    elif args.model == 'IMAGEONLY':    
        print('Loading image only model ...')
        from models import IMAGEONLY
        model = IMAGEONLY(args)
        if args.image_only_model == 'resnet152':
            transform = 'default'
        elif args.image_only_model == 'clip':
            transform = model.preprocess
        else:
            raise ValueError(f"Image only model encoder {args.image_only_model_encoder} not implemented.")

    elif args.model == 'TEXTONLY':
        print('Loading text only model ...')
        from models import TEXTONLY
        model = TEXTONLY(args)
        transform = model.preprocess

    elif args.model == 'BLIPv2':
        print('Loading BLIPv2 model ...')
        from models import BLIPv2
        model = BLIPv2(args)
        transform = model.match_vis_processors['eval']
    
    elif args.model == 'instructBLIP':
        print('Loading instructBLIP model ...')
        from models import instructBLIP
        model = instructBLIP(args)

        transform = model.vis_processors['eval']
    elif args.model == 'BLIP4CIR':  # train blip2 in clip4cir method
        print('Loading BLIP4CIR model ...')
        from models import BLIP4CIR
        model = BLIP4CIR(args)

        transform = model.preprocess
    elif args.model == 'RerankFile':
        print('Loading rerank result from file ...')
        from models import TEXTONLY
        model = TEXTONLY(args)
        transform = model.preprocess
    elif args.model == 'Pic2word':
        print('Loading Pic2word model ...')
        from models import Pic2word
        model = Pic2word(args)
        transform = model.preprocess        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total params:',pytorch_total_params)

        
    elif args.model == 'SEARLE':
        print('Loading SEARLE model ...')
        from models import SEARLE
        model = SEARLE(args)
        transform = model.preprocess
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total params:',pytorch_total_params)
        
    elif 'OPENCLIP4CIR' in args.model:
        print('Loading OPENCLIP4CIR model ...')
        from models import OPENCLIP4CIR
        model = OPENCLIP4CIR(args)
        transform = model.preprocess
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total params:',pytorch_total_params)
    elif args.model == 'BIBLIP4CIR':
        print('Loading BIBLIP4CIR model ...')
        from models import BIBLIP4CIR
        model = BIBLIP4CIR(args)
        transform = model.preprocess
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Total params:',pytorch_total_params)

    elif args.model == 'SPRC':
        print('Loading SPRC model ...')
        from models import SPRC
        model = SPRC(args)
        transform = model.preprocess

    else:
        raise ValueError(f"Model {args.model} not implemented.")
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
    return args, model, vocab, transform


if __name__ == '__main__':

    parser = arg_parser()
    args = verify_input_args(parser.parse_args())
    # Load model & vocab
    args, model, vocab,transform = load_model(args)
    start = time.time()
    with torch.no_grad():
        print('current split:',args.studied_split)
        message, _ = validate(model, args, vocab, transform = transform, split = args.studied_split)
    print(message)

    # save printed message on .txt file

    basename = ""
    if os.path.basename(args.ckpt) != "model_best.pth":
        basename = "_%s" % os.path.basename(os.path.basename(args.ckpt)).split(".")[0]

    save_txt = os.path.abspath( os.path.join(args.ckpt, os.path.pardir, 'eval_message%s.txt' % basename) )
    if not os.path.exists( os.path.dirname(save_txt)):
        os.makedirs(os.path.dirname(save_txt))
        print('build folder:',os.path.dirname(save_txt))
    print('results saved in:', save_txt)

    with open(save_txt, 'a') as f:
        f.write(args.data_name + ' ' + args.studied_split + ' ' + args.exp_name + '\n######')
        f.write(message + '\n######\n')

    end = time.time()
    print("\nProcessing time : ", end - start)
