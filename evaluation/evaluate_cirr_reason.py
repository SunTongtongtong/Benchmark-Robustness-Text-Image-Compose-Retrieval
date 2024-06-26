import dataset as data
import json
import os
import torch



def deal_with_CIRR_reason(args, vocab, cs_sorted_ind, split):
	soft_targets = get_data_in_list(data.get_soft_targets_loader(args, vocab, split))

	rK = eval_for_CIRR_reason(cs_sorted_ind,  soft_targets, args.recall_k_values)

	message = "\n\n>> EVALUATION <<"
	message += results2string(rK, "R")
	return message, 0
# , val_mes


def results2string(values, metric_name):
	message = ""
	for k, v in values:
		message += ("\nMetric {}@%d: %.2f" % (k, v)).format(metric_name)
	return message


def get_data_in_list(data_loader):
	L = []
	for data in data_loader:
		d, _ = data # skip the dataset index ; is already a list (batch size)
		if type(d) is tuple:
			L += list(d)
		else: # torch.tensor
			L += d.tolist()
	return L


def eval_for_CIRR_reason(cs_sorted_ind,  soft_targets,
					recall_k_values):

	"""
	Input:
		cs_sorted_ind: torch.tensor size (nb of queries, retrieved nb of candidate
			targets), containing the indices of the top ranked candidate targets
			for each query.
		subset_ids: list of size (nb of queries). Its sublists contain the
			indices of a subset of potential targets to compare to, for each query,
			as a specific metric, R_subset@K.
		soft_targets: list of size (nb of queries). It contains one dictionary
			for each query, such that {soft target id: soft target qualification},
			indicating what candidate target (soft target id) can be considered
			as acceptable target (soft target qualification: 1 is OK, 0.5 is 50%
			OK, and -1 is not OK) for the given query.

	More info on the CIRR Github project page:
	https://github.com/Cuberick-Orion/CIRR
	
	"""

	out_rK = []
	# Recall@K
	for k in recall_k_values:
		r = 0.0
		for i, nns in enumerate(cs_sorted_ind):
			highest_r = 0.0
			# for ii,ss in soft_targets[i]:
			ii = soft_targets[i]
			if ii in nns[:k]:
				highest_r = 1 # update the score
			r += highest_r
		r /= len(cs_sorted_ind)
		out_rK += [(k, r*100)]

	return out_rK


def dict_to_json(d, filepath):
	with open(filepath, "w") as file:
		json.dump(d, file)
	return "File saved at {}".format(filepath)