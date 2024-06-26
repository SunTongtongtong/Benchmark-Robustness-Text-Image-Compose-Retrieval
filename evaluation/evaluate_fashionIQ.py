import dataset as data
import json
import os
import torch




def deal_with_fashionIQ(args, vocab, cs_sorted_ind, split):
	name_categories = ["dress", "shirt", "toptee"]
	results=[]
	for idx, category in enumerate(name_categories):
		args.categories = category
		query_loader, targets_loader = data.get_eval_loaders(args, vocab, split)
		dataset = query_loader.dataset
		ranking_result = cs_sorted_ind[idx]
		ret = torch.zeros(len(query_loader.dataset), requires_grad=False)
		for idx,single_query_img in enumerate(ranking_result):
			index=idx // 2
			ann = dataset.annotations[index]
			img_trg_id = [dataset.image_id2name.index(ann['target'])]
			ret[idx]=get_rank_of_GT(torch.tensor(single_query_img), img_trg_id)[0]
		results.append(ret)
	message, val_mes = results_func(results, args)
	return message, val_mes



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
        categories = ['dress', 'shirt', 'toptee']
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

