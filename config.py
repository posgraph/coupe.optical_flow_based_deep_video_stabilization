from easydict import EasyDict as edict
import json
import warp
import numpy as np
import os

sp_config = edict()
config = edict()
config.TRAIN = edict()
config.TEST = edict()

# config
config.feature_names = 'mean_rgb'
config.feature_sizes = 1024
config.frame_features = True

# Adam
config.batch_size = 1	# 1
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

# learning rate
config.TRAIN.n_epoch = 10000
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 20

# Dataset
offset = '/data1/DeepStabV2similarityHomo/'
config.TRAIN.stab_path = offset + 'train/stab_similarity_frame/'
config.TRAIN.unstab_path = offset + 'train/unstab_similarity_frame/'
config.TRAIN.homo_path = offset + 'train/homo/'
#offset = '/data1/junyonglee/video_stab/train'
#config.TRAIN.stab_path = os.path.join(offset, 'stab_similarity_frames_NR')
#config.TRAIN.unstab_path = os.path.join(offset, 'unstab_similarity_frames_NR')
#config.TRAIN.homo_path = offset + '/homo/'

## test set location
config.TEST.stab_path = offset + 'test/stab_similarity_frame/'
config.TEST.unstab_path = offset + 'test/unstab_similarity_frame/'
config.TEST.unstab_path_testdataset = offset + 'test/unstab_testset/'
config.TEST.homo_path = offset + 'test/homo/'
#offset = '/data1/junyonglee/video_stab/test'
#config.TEST.stab_path = os.path.join(offset, 'stab_similarity_frames_NR')
#config.TEST.unstab_path = os.path.join(offset, 'unstab_similarity_frames_NR')
#config.TEST.unstab_path_testdataset = '/data1/junyonglee/video_stab/eval/unstab'
#config.TEST.homo_path = offset + '/homo/'

## train image size
config.height = 352 # (32 * 11)
config.width = 352

## LSTM
# train sequence length
config.seq_length = 1 # 10
# number feature dim of cell
config.num_features = 256

# a number of previous output
config.prev_output_num = 10 #15
config.prev_output_num_weight = [1,0.9139,0.8353,0.7634,0.6977,0.6376,0.5827,0.5326,0.4868,0.4449]
#								[1,0.99,0.961,0.914,0.852, \
#								0.779,0.700,0.613,0.527,0.445,\
#								0.368,0.298,0.237,0.185,0.141,\
#								0.105,0.077,0.056,0.039,0.027,\
#								0.018,0.012,0.008,0.005,0.003,\
#								0.002,0.002,0.002,0.002,0.002]
#[1,0.9139,0.8353,0.7634,0.6977,0.6376,0.5827,0.5326,0.4868,0.4449]

## log & checkpoint & samples
# every global step
config.TRAIN.write_log_every = 50
config.TRAIN.write_image_every = 250
config.TRAIN.write_ckpt_every = 1000
config.TRAIN.write_sample_every = 1000
# every epoch
config.TRAIN.refresh_image_log_every = 20

# save dir
# offset = '/Mango/Users/JunyongLee/hub/logs/'
offset = '/Jarvis/logs/LJH/'
config.root_dir = offset + 'deep_video_stabilization/'


# salient ckpt dir
config.TRAIN.sal_ckpt_dir = config.root_dir + 'salient_detector/'

config.TEST.save_path = config.root_dir + 'result/'

## Spatial Transformer
sp_config.grid_size = 11
sp_config.batch_size = config.batch_size
sp_config.out_size = [config.width, config.height]

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')
