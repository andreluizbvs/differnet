'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda' # or 'cpu'
import torch
torch.cuda.set_device(0)

# data settings
# dataset_path = "vari-grip"
dataset_path = "isolador_vidro"
# class_name = "corrosion"
class_name = "missing_cap"
modelname = dataset_path
image_folder = dataset_path+'/'+class_name+'/test/good'

# model setting
MODEL_DIR = './models'+'/'+modelname
WEIGHT_DIR = './weights'+'/'+modelname

img_size = (512, 512)
img_dims = [3] + list(img_size)

# metric

metric = 'F1-Score'

# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 24 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size * n_transforms // n_transforms_test

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 24
sub_epochs = 8

# output settings
verbose = True
grad_map_viz = False
hide_tqdm_bar = True
hide_tqdm_bar_test = False
save_model = True
save_roc = True
save_prc = True
save_hist = True
