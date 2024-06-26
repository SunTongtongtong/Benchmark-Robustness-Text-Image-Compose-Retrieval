import argparse
import os
from config import MAIN_DIR, VOCAB_DIR, CKPT_DIR, RANKING_DIR, HEATMAP_DIR

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')
    # model definition
    parser.add_argument('--model', default='maaf', type=str, help='backbone network')
    
    parser.add_argument('--data_name', default='fashioniq', help='which dataset.')

    # shitong: for corruption
    parser.add_argument('--img_corrupt',default=' ',type=str,help='type of corruption on image')
    parser.add_argument('--txt_corrupt',default=' ',type=str,help='type of corruption on text')
    parser.add_argument('--img_corrupt_level',default=0,type=int, help='severity of corruption on image')   


    parser.add_argument('--validate', default='val', choices=('val', 'test', 'test-val'), help='Split(s) on which the model should be validated (if 2 are given, 2 different checkpoints of     model_best will be kept, one for each validating split).')
    parser.add_argument('--studied_split', default="val", help="Split to be used for the computation (this does not impact the usual training & validation pipeline, but impacts other scripts (for evaluation or visualizations purposes)).")
    parser.add_argument('--exp_name', default='X', help='Experiment name, used as sub-directory to save experiment-related files (model, ranking files, heatmaps...).')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='Path of the ckpt to resume from (default: none).')
    parser.add_argument('--ckpt_dir', default=CKPT_DIR, help='Directory in which to save the models from the different experiments.')
    parser.add_argument('--ranking_dir', default=RANKING_DIR, type=str, help='Directory in which to save the ranking/prediction files, if any to save.')
    parser.add_argument('--wemb_type', default='glove', choices=('glove', 'None'), type=str, help='Word embedding (glove|None).')
    parser.add_argument('--vocab_dir', default=VOCAB_DIR, help='Path to saved vocabulary pickle files')
    parser.add_argument('--embed_dim', default=512, type=int, help='Dimensionality of the final text & image embeddings.')
    parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.')
    parser.add_argument('--txt_enc_type', default='bigru', choices=('bigru', 'lstm'), help="The text encoder (bigru|lstm).")
    parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embeddings.')
    parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image encoder.')

    parser.add_argument('--load_image_feature', default=0, type=int, help="Whether (if int > 0) to load pretrained image features instead of loading raw images and using a cnn backbone. Indicate  the size of the feature (int).")
    parser.add_argument('--gradcam', action='store_true', help='Keep gradients & activations computed while encoding the images to further interprete what the network uses to make its decision.')
    parser.add_argument('--cnn_type', default='resnet50', help='The CNN used as image encoder.')
    parser.add_argument('--temperature', default=2.65926, type=float, help='Temperature parameter.')
    parser.add_argument('--categories', default='all', type=str, help='Names of the data categories to consider for a given dataset. Category names must be separated with a space. Specify "all" to consider them all (the interpretation of "all" depends on the dataset).')

    parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of a mini-batch.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')

    parser.add_argument('--lstm_hidden_dim', default=512, type=int, help='Number of hidden units in the LSTM.')


    # TIRG and ARTEMIS


    # for MAAF
    parser.add_argument('--image_model_arch', type=str, default='resnet50')
    parser.add_argument('--image_model_path', type=str, default='')
    parser.add_argument(
        '--freeze_text_model', action='store_true',
        help='If added the loaded text model weights will not be finetuned')
    parser.add_argument(
        '--freeze_img_model', action='store_true',
        help='If added the loaded image model weights will not be finetuned')
    parser.add_argument('--att_layer_spec', type=str, default="34")
    parser.add_argument('--text_model_arch', type=str, default='lstm')
    parser.add_argument('--text_model_layers', type=int, default=1)
    parser.add_argument('--threshold_rare_words', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--number_attention_blocks', type=int, default=2)
    parser.add_argument('--width_per_attention_block', type=int, default=128)
    parser.add_argument('--number_attention_heads', type=int, default=8)
    parser.add_argument('--attn_positional_encoding', default=None)
    parser.add_argument('--attn_softmax_replacement', type=str, default="none")
    parser.add_argument('--sequence_concat_img_through_attn', action="store_false",
        help="target image pathway goes through embedding layers")
    parser.add_argument('--resolutionwise_pool', action='store_false')
    parser.add_argument('--sequence_concat_include_text', action="store_false",
        help="use post-attn text embeddings in pooling to get final composed embedding")

    # CIRPLANT
    parser.add_argument('--task_name', default='cirr', help='which dataset.')
    parser.add_argument("--max_seq_length", default=40, type=int,
                      help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_img_seq_length", default=1, type=int, help="The maximum total input image sequence length.")

    # reasoning
    parser.add_argument('--reasoning_type', default='none', help='which dataset.') # for reasoning caption file name 

    #image only 
    parser.add_argument('--image_only_model', default='clip', help='which dataset.') 
	
    # for heatmap
    parser.add_argument('--heatmap_dir', default=HEATMAP_DIR, type=str, help='Directory in which to save the heatmaps.')

    #for pic2word
    parser.add_argument('--tcpurl', default='tcp://127.0.0.1:6101', help='which dataset.')


    #for FashionVil
    # parser.add_argument('--opts', nargs=argparse.REMAINDER, default=["config=projects/fashionvil/configs/e2e_composition.yaml","model=fashionvil","dataset=fashioniq,","run_type=test","checkpoint.resume_file=./save/fashionvil_composition_fashioniq_e2e_pretrain_final/fashionvil_comp_final.pth"])
    # parser.add_argument('--config', type=str, default="./models/fashionvil/e2e_composition.yaml")
    # parser.add_argument('--dataset', default='fashioniq', help='which dataset.')
    # parser.add_argument('--run_type', default='test', help='which dataset.')
    # parser.add_argument(
    #         "-co",
    #         "--config_override",
    #         type=str,
    #         default=None,
    #         help="Use to override config from command line directly",
    #     )

    # shitong: maybe have evaluation metrc later    

    # parser.add_argument('-d', '--depth', default=18, type=int, metavar='N',
    #                     help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152])
    # parser.add_argument('--dropout', default=0.5, type=float,
    #                     help='dropout ratio before the final layer')
    # parser.add_argument('--groups', default=16, type=int, help='number of frames')
    # parser.add_argument('--frames_per_group', default=1, type=int,
    #                     help='[uniform sampling] number of frames per group; '
    #                          '[dense sampling]: sampling frequency')
    # parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
    #                     help='skip the temporal pooling in the model')
    # parser.add_argument('--pooling_method', default='max',
    #                     choices=['avg', 'max'], help='method for temporal pooling method')
    # parser.add_argument('--dw_t_conv', dest='dw_t_conv', action='store_true',
    #                     help='[S3D model] only enable depth-wise conv for temporal modeling')
    # # model definition: temporal model for 2D models
    # parser.add_argument('--temporal_module_name', default=None, type=str,
    #                     help='[2D model] which temporal aggregation module to use. None == TSN',
    #                     choices=[None, 'TSN', 'TAM'])
    # parser.add_argument('--blending_frames', default=3, type=int, help='For TAM only.')
    # parser.add_argument('--blending_method', default='sum',
    #                     choices=['sum', 'max'], help='method for blending channels in TAM')
    # parser.add_argument('--no_dw_conv', dest='dw_conv', action='store_false',
    #                     help='[2D model] disable depth-wise conv for TAM')
    # parser.add_argument(
    #     "--cfg",
    #     dest="cfg_file",
    #     help="Path to the config file",
    #     default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
    #     type=str,
    # )

    # parser.add_argument('--tsm', action='store_true',
    #                     help='adding tsm module.')

    # # training setting
    # parser.add_argument('--cuda',help='set GPUs to use',type=str,default='0')
    # parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    # parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
    #                     help='Disable cudnn to search the best mode (avoid OOM)')
    # parser.add_argument('-b', '--batch-size', default=256, type=int,
    #                     metavar='N', help='mini-batch size (default: 256)')
    # parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
    #                     metavar='LR', help='initial learning rate')
    # parser.add_argument('--lr_scheduler', default='cosine', type=str,
    #                     help='learning rate scheduler',
    #                     choices=['step', 'multisteps', 'cosine', 'plateau'])
    # parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
    #                     metavar='LRSteps', help='[step]: use a single value: the periodto decay '
    #                                             'learning rate by 10. '
    #                                             '[multisteps] epochs to decay learning rate by 10')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    # parser.add_argument('--nesterov', action='store_true',
    #                     help='enable nesterov momentum optimizer')
    # parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    #                     metavar='W', help='weight decay (default: 5e-4)')
    # parser.add_argument('--epochs', default=50, type=int, metavar='N',
    #                     help='number of total epochs to run')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    # parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
    #                     help='use pre-trained model')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='manual epoch number (useful on restarts)')
    # parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
    #                     help='clip the total norm of gradient before update parameter')
    # parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
    #                     action='store_false',
    #                     help='disable to load imagenet model')

    # # data-related
    # parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
    #                     help='number of data loading workers (default: 18)')
    # parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')

    # parser.add_argument('--threed_data', action='store_true',
    #                     help='format data to 5D for 3D onv.')
    # parser.add_argument('--input_size', default=224, type=int, metavar='N', help='spatial size')
    # parser.add_argument('--disable_scaleup', action='store_true',
    #                     help='do not scale up and then crop a small region, '
    #                          'directly crop the input_size from center.')
    # parser.add_argument('--random_sampling', action='store_true',
    #                     help='[Uniform sampling only] perform non-deterministic frame sampling '
    #                          'for data loader during the evaluation.')
    # parser.add_argument('--dense_sampling', action='store_true',
    #                     help='perform dense sampling for data loader')
    # parser.add_argument('--augmentor_ver', default='v1', type=str,
    #                     help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    # parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
    #                     metavar='scale_range', help='scale range for augmentor v2')
    # parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
    #                     choices=['rgb', 'flow'])
    # parser.add_argument('--mean', type=float, nargs="+", metavar='MEAN',
    #                     help='[Data normalization] mean, dimension should be 3 for RGB, 1 for flow')
    # parser.add_argument('--std', type=float, nargs="+", metavar='STD',
    #                     help='[Data normalization] std, dimension should be 3 for RGB, 1 for flow')
    # parser.add_argument('--dataset_path', type=str,
    #                     help='the path of dataset')
    # # logging
    # parser.add_argument('--logdir', default='', type=str, help='log path')
    # parser.add_argument('--print-freq', default=100, type=int,
    #                     help='frequency to print the log during the training')
    # parser.add_argument('--show_model', action='store_true',
    #                     help='show model and then exit intermediately')

    # # for testing and validation
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10],
    #                     help='[Test.py only] number of crops.')
    # parser.add_argument('--num_clips', default=1, type=int,
    #                     help='[Test.py only] number of clips.')


    # # for distributed learning, not supported yet
    # parser.add_argument('--sync-bn', action='store_true',
    #                     help='sync BN across GPUs')
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=0, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')
    # parser.add_argument('--size_level',default = 'b',type = str,help = 'transformer size level',choices = [
    #     't','s','b','l'])

    # parser.add_argument('--warm_up_epoch',default = 3,type = int,help = 'warm up epochs')
    # parser.add_argument('--weight_decay',default=0.02,type=float,help='weigth decay')
    return parser


def verify_input_args(args):
	"""
	Check that saving directories exist (or create them).
	Define default values for each dataset.
	"""

	############################################################################
	# --- Check that directories exist (or create them)

	# training ckpt directory
	ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
	if args.data_name == 'cirr': # creade submission folder only for cirr
		if not os.path.isdir(args.ckpt_dir):
			print('Creating a directory: {}'.format(ckpt_dir))
			os.makedirs(ckpt_dir)

	    # validated ckpt directories
		args.validate = args.validate.split('-') # splits for validation
		for split in args.validate:
			ckpt_val_dir = os.path.join(ckpt_dir, split)
		if not os.path.isdir(ckpt_val_dir):
			print('Creating a directory: {}'.format(ckpt_val_dir))
			os.makedirs(ckpt_val_dir)

	# prediction directory
	if not os.path.isdir(args.ranking_dir):
		os.makedirs(args.ranking_dir)

	############################################################################
	# --- Process input arguments: deduce some new arguments from provided ones.

	if args.wemb_type == "None":
		args.wemb_type = None

	# Number and name of data categories
	args.name_categories = [None]
	args.recall_k_values = [1, 10, 50]
	args.recall_subset_k_values = None
	args.study_per_category = False # to evaluate the model on each category separately (case of a dataset with multiple categories)
	if args.data_name == 'fashionIQ':
		args.name_categories = ["dress", "shirt", "toptee"]
		args.recall_k_values = [10, 50]
		args.study_per_category = True
	elif args.data_name == 'cirr':
		args.recall_k_values = [1, 5, 10, 50,80,100] # shitong: change add 80 100
		args.recall_subset_k_values = [1, 2, 3]
	else:
		args.recall_k_values = [1, 5, 10, 50]
	args.number_categories = len(args.name_categories)
		
	return args