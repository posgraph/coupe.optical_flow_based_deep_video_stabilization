from config import sp_config, config, log_config
from utils import *
from model import *
from spatial_transformer import ProjectiveSymmetryTransformer, ProjectiveTransformer, AffineSymmetryTransformer,SimilarityTransformer
from tensorlayer.layers import *
import NLDF
from cell import ConvLSTMCell, ConvGRUCell
import os
from scipy import ndimage
import scipy

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from random import shuffle
import matplotlib
import datetime
import time
import shutil
from numpy.linalg import inv
import cv2

import abc
import os
import uuid
import random
from PIL import Image

import collections
from data_loader import Data_Loader

slim = tf.contrib.slim

batch_size = 10 # config.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.ceil(np.sqrt(batch_size)))

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_sizei = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_sizei)
    batch_idx = tf.reshape(batch_idx, (batch_sizei, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def tf_warp(img, flow, H, W):
    flow = tf.transpose(flow, [0, 3, 1, 2])
    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,0)

    y = tf.expand_dims(y,0)
    y = tf.expand_dims(y,0)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    
    grid  = tf.concat([x,y],axis = 1)
    flows = grid+flow
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:,0,:,:]
    y = flows[:,1,:,:]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0,  tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)


    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out

def grayscale(input):
    output = np.zeros( [128,128,config.prev_output_num])
    for i in range(config.prev_output_num):
        output[:,:,i] = color.rgb2gray(cv2.resize(np.squeeze(input[i,:,:,:]),dsize=(128,128),interpolation=cv2.INTER_CUBIC) )
    return output

def train():
    ## CREATE DIRECTORIES
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])

    ckpt_dir = mode_dir + '/checkpoint'
    init_dir = mode_dir + '/init'
    log_dir_scalar = mode_dir + '/log/scalar'
    log_dir_image = mode_dir + '/log/image'
    sample_dir = mode_dir + '/samples/0_train'
    config_dir = mode_dir + '/config'
    img_dir = mode_dir + '/img'

    if tl.global_flag['delete_log']:
        shutil.rmtree(ckpt_dir, ignore_errors = True)
        shutil.rmtree(log_dir_scalar, ignore_errors = True)
        shutil.rmtree(log_dir_image, ignore_errors = True)
        shutil.rmtree(sample_dir, ignore_errors = True)
        shutil.rmtree(config_dir, ignore_errors = True)
        shutil.rmtree(img_dir, ignore_errors = True)

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(log_dir_scalar)
    tl.files.exists_or_mkdir(log_dir_image)
    tl.files.exists_or_mkdir(sample_dir)
    tl.files.exists_or_mkdir(config_dir)
    tl.files.exists_or_mkdir(img_dir)
    log_config(config_dir, config)

    ## DEFINE SESSION
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
    #skip_length, stab_path, unstab_path, height, width, batch_size, 
    
    data_loader = Data_Loader([1, 9, 17, 25, 28, 29, 30, 31, 32], config.TRAIN.stab_path, config.TRAIN.unstab_path, 384, 512, batch_size, is_train = True, thread_num = 3)
    data_loader_test = Data_Loader([1, 9, 17, 25, 28, 29, 30, 31, 32], config.TEST.stab_path, config.TEST.unstab_path, 384, 512, batch_size, is_train = False, thread_num = 2)
    
    ## DEFINE MODEL
    # input
    inputs = collections.OrderedDict()
    with tf.variable_scope('input'):

        #patches_in = tf.placeholder('float32', [batch_size, sample_num, h, w, 3], name = 'input_frames')
        inputs['unstab_image'] = tf.placeholder('float32', [None, 384, 512, 3], name = 'input_unstab')
        inputs['gtstab_image'] = tf.placeholder('float32', [None, 384, 512, 3], name = 'input_gtstab')
        inputs['stab_image'] = tf.placeholder('float32', [None, 384, 512, 3*8], name = 'input_stab')

    with tf.variable_scope('main_net') as scope:
        outputs = flownetS_pyramid( tf.concat([inputs['stab_image'],inputs['unstab_image']],3) ,batch_size,is_train=True )
        outputs_test = flownetS_pyramid(tf.concat([inputs['stab_image'],inputs['unstab_image']],3),batch_size,is_train=False,reuse=True )


    def masked_MSE(pred, gt, mask):
        pred_mask = pred * mask
        gt_mask = gt * mask

        MSE = tf.reduce_sum(tf.squared_difference(pred_mask, gt_mask), axis = [1, 2, 3])

        safe_mask = tf.cast(tf.where(tf.equal(mask, tf.zeros_like(mask)), mask + tf.constant(1e-8), mask), tf.float32)
        MSE = MSE / tf.reduce_sum(safe_mask, axis = [1, 2, 3])
        
        # MSE = tf.div_no_nan(MSE, tf.reduce_sum(mask, axis = [1, 2, 3]))
        return tf.reduce_mean(MSE)

    def lossterm(predict_flow,stab_image,unstab_image):
        size = [predict_flow.shape[1], predict_flow.shape[2]]
        downsampled_stab = tf.image.resize_images(stab_image, size )#downsample(gt_flow, size)
        downsampled_unstab = tf.image.resize_images(unstab_image, size )#downsample(gt_flow, size)
        warpedimg = tf_warp(downsampled_unstab, predict_flow, predict_flow.shape[1], predict_flow.shape[2])
        #loss6 = tl.cost.mean_squared_error(downsampled_stab6, warpedimg6, is_mean = True, name = 'loss6') #* 0.32
        mask = tf_warp(tf.ones_like(downsampled_unstab), predict_flow, predict_flow.shape[1], predict_flow.shape[2])
        print(warpedimg)
        print(downsampled_stab)
        print(mask)
        return masked_MSE(warpedimg, downsampled_stab, mask),warpedimg

        #size = [stab_image.shape[1], stab_image.shape[2]]
        #flowsize = [predict_flow.shape[1],predict_flow.shape[2]]
        #multiplyConst = tf.cast(tf.shape(stab_image)[1],tf.float32)/tf.cast(tf.shape(predict_flow)[1],tf.float32) 
        #upsampled_flow = tf.image.resize_images( tf.multiply(predict_flow,multiplyConst), size ) #downsample(gt_flow, size)
        #warpedimg = tf_warp(unstab_image, upsampled_flow, stab_image.shape[1], stab_image.shape[2])
        ##loss6 = tl.cost.mean_squared_error(downsampled_stab6, warpedimg6, is_mean = True, name = 'loss6') #* 0.32
        #mask = tf_warp(tf.ones_like(unstab_image), upsampled_flow, stab_image.shape[1], stab_image.shape[2])
        #print(upsampled_flow )
        #print(warpedimg)
        #print(mask)
        #return masked_MSE(warpedimg, stab_image, mask),warpedimg


    ## DEFINE LOSS
    with tf.variable_scope('loss'):

        loss6,warpedimg6 = lossterm(outputs['predict_flow6'],inputs['gtstab_image'],inputs['unstab_image'])
        loss5,warpedimg5 = lossterm(outputs['predict_flow5'],inputs['gtstab_image'],inputs['unstab_image'])
        loss4,warpedimg4 = lossterm(outputs['predict_flow4'],inputs['gtstab_image'],inputs['unstab_image'])
        loss3,warpedimg3 = lossterm(outputs['predict_flow3'],inputs['gtstab_image'],inputs['unstab_image'])
        loss2,warpedimg2 = lossterm(outputs['predict_flow2'],inputs['gtstab_image'],inputs['unstab_image'])

        loss6_1,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss5_1,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss4_1,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss3_1,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss2_1,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss_1 = (loss6_1+loss5_1+loss4_1+loss3_1+loss2_1)*np.exp(-1/3)

        loss6_2,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss5_2,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss4_2,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss3_2,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss2_2,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss_2 = (loss6_2+loss5_2+loss4_2+loss3_2+loss2_2)*np.exp(-2/3)

        loss6_3,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss5_3,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss4_3,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss3_3,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss2_3,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss_3 = (loss6_3+loss5_3+loss4_3+loss3_3+loss2_3)*np.exp(-3/3)

        loss6_4,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss5_4,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss4_4,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss3_4,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss2_4,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss_4 = (loss6_4+loss5_4+loss4_4+loss3_4+loss2_4)*np.exp(-4/3)

        loss6_5,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss5_5,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss4_5,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss3_5,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss2_5,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss_5 = (loss6_5+loss5_5+loss4_5+loss3_5+loss2_5)*np.exp(-7/3)
        
        var6 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow6']))*2e-8*3
        var5 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow5']))*2e-8*3
        var4 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow4']))*2e-8*3
        var3 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow3']))*4e-8*1.5
        var2 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow2']))*4e-8*1.5

        loss_main = tf.identity( loss6+loss5+loss4+loss3+loss2+var6+var5+var4+var3+var2, name = 'total')
            
    with tf.variable_scope('loss_test'):

        loss6,warpedimg6 = lossterm(outputs_test['predict_flow6'],inputs['gtstab_image'],inputs['unstab_image'])
        loss5,warpedimg5 = lossterm(outputs_test['predict_flow5'],inputs['gtstab_image'],inputs['unstab_image'])
        loss4,warpedimg4 = lossterm(outputs_test['predict_flow4'],inputs['gtstab_image'],inputs['unstab_image'])
        loss3,warpedimg3 = lossterm(outputs_test['predict_flow3'],inputs['gtstab_image'],inputs['unstab_image'])
        loss2,warpedimg2 = lossterm(outputs_test['predict_flow2'],inputs['gtstab_image'],inputs['unstab_image'])

        loss6_1,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss5_1,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss4_1,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss3_1,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss2_1,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,21:24],inputs['unstab_image'])
        loss_1 = (loss6_1+loss5_1+loss4_1+loss3_1+loss2_1)*np.exp(-1/3)

        loss6_2,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss5_2,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss4_2,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss3_2,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss2_2,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,18:21],inputs['unstab_image'])
        loss_2 = (loss6_2+loss5_2+loss4_2+loss3_2+loss2_2)*np.exp(-2/3)

        loss6_3,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss5_3,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss4_3,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss3_3,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss2_3,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,15:18],inputs['unstab_image'])
        loss_3 = (loss6_3+loss5_3+loss4_3+loss3_3+loss2_3)*np.exp(-3/3)

        loss6_4,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss5_4,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss4_4,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss3_4,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss2_4,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,12:15],inputs['unstab_image'])
        loss_4 = (loss6_4+loss5_4+loss4_4+loss3_4+loss2_4)*np.exp(-4/3)

        loss6_5,_ = lossterm(outputs['predict_flow6'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss5_5,_ = lossterm(outputs['predict_flow5'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss4_5,_ = lossterm(outputs['predict_flow4'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss3_5,_ = lossterm(outputs['predict_flow3'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss2_5,_ = lossterm(outputs['predict_flow2'],inputs['stab_image'][:,:,:,9:12],inputs['unstab_image'])
        loss_5 = (loss6_5+loss5_5+loss4_5+loss3_5+loss2_5)*np.exp(-7/3)

        var6 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow6']))*2e-8*3
        var5 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow5']))*2e-8*3
        var4 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow4']))*2e-8*3
        var3 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow3']))*4e-8*1.5
        var2 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow2']))*4e-8*1.5

        loss_main_test = tf.identity( loss6+loss5+loss4+loss3+loss2+var6+var5+var4+var3+var2 , name = 'totaltest')
    ## DEFINE OPTIMIZER
    # variables to save / train
    main_vars = tl.layers.get_variables_with_name('main_net', True, False)
    save_vars = tl.layers.get_variables_with_name('main_net', False, False)

    # define optimizer
    with tf.variable_scope('Optimizer'):
        learning_rate = tf.Variable(lr_init, trainable = False)
        optim_main = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss_main, var_list = main_vars)


    ## DEFINE SUMMARY
    # writer
    writer_scalar = tf.summary.FileWriter(log_dir_scalar, sess.graph, flush_secs=30, filename_suffix = '.loss_log')
    writer_image = tf.summary.FileWriter(log_dir_image, sess.graph, max_queue=100, flush_secs=30, filename_suffix = '.image_log')
 
    # for train
    loss_sum_main_list = []
    with tf.variable_scope('loss'):
        loss_sum_main_list.append(tf.summary.scalar('total', loss_main))
        loss_sum_main_list.append(tf.summary.scalar('loss6', loss6))
        loss_sum_main_list.append(tf.summary.scalar('loss5', loss5))
        loss_sum_main_list.append(tf.summary.scalar('loss4', loss4))
        loss_sum_main_list.append(tf.summary.scalar('loss3', loss3))
        loss_sum_main_list.append(tf.summary.scalar('loss2', loss2))
        loss_sum_main_list.append(tf.summary.scalar('var6', var6))
        loss_sum_main_list.append(tf.summary.scalar('var5', var5))
        loss_sum_main_list.append(tf.summary.scalar('var4', var4))
        loss_sum_main_list.append(tf.summary.scalar('var3', var3))
        loss_sum_main_list.append(tf.summary.scalar('var2', var2))
    loss_sum_main = tf.summary.merge(loss_sum_main_list)

    loss_sum_test_list = []
    with tf.variable_scope('loss_test'):
        loss_sum_test_list.append(tf.summary.scalar('totaltest', loss_main_test))
    loss_sum_test_main = tf.summary.merge(loss_sum_test_list)    

    image_sum_list = []
    image_sum_list.append(tf.summary.image('unstab_image', fix_image_tf(inputs['unstab_image'], 1)))
    image_sum_list.append(tf.summary.image('stab_image', fix_image_tf(inputs['gtstab_image'], 1)))
    image_sum_list.append(tf.summary.image('warpedimg2', fix_image_tf(warpedimg2, 1)))
    image_sum_list.append(tf.summary.image('warpedimg3', fix_image_tf(warpedimg3, 1)))
    image_sum_list.append(tf.summary.image('warpedimg4', fix_image_tf(warpedimg4, 1)))
    image_sum_list.append(tf.summary.image('warpedimg5', fix_image_tf(warpedimg5, 1)))
    image_sum_list.append(tf.summary.image('warpedimg6', fix_image_tf(warpedimg6, 1)))
    image_sum = tf.summary.merge(image_sum_list)


    ## INITIALIZE SESSION
    tl.layers.initialize_global_variables(sess)
    #flownetSaver = tf.train.Saver(tl.layers.get_variables_with_name('FlowNetSD'))
    #flownetSaver.restore(sess, './flownet/checkpoints/FlowNetSD/flownet-SD.ckpt-0')
    tl.files.load_and_assign_npz_dict(name = '/Jarvis/logs/LJH/deep_video_stabilization/main_flownetS_pyramid_noprevloss_dataloader/fixed_ckpt/main_flownetS_pyramid_noprevloss_dataloader_prevloss_29k.npz', sess = sess)

    ## START TRAINING
    print('*****************************************')
    print('             TRAINING START')
    print('*****************************************')
    data_loader.init_data_loader(inputs)
    data_loader_test.init_data_loader(inputs)
    global_step = 0
    for epoch in range(0, n_epoch + 1):
        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(learning_rate, lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(learning_rate, lr_init))

        idx = 0
        while 1:
            step_time = time.time()

            print('==============================================================')

            feed_dict, is_end, is_not_batchsize = data_loader.feed_the_network()
            if is_end: break;
            if is_not_batchsize: continue;
            _ = sess.run(optim_main, feed_dict)

            # print(status)
            err_main, lr = \
            sess.run([loss_main, learning_rate], feed_dict)
            print('[%s] global step: %d  %4d/%4d time: %4.2fs, err[main: %1.2e], lr: %1.2e' % \
                (tl.global_flag['mode'],global_step, idx, data_loader.num_itr, time.time() - step_time, err_main, lr))
            ## SAVE LOGS
            # save loss & image log
            if global_step % config.TRAIN.write_log_every == 0:
                summary_loss  = sess.run(loss_sum_main, feed_dict)
                summary_image  = sess.run(image_sum, feed_dict)
                writer_scalar.add_summary(summary_loss, global_step)
                writer_image.add_summary(summary_image, global_step)
                print('save loss log')
            # save checkpoint
            if global_step != 0 and global_step % config.TRAIN.write_ckpt_every == 0:
                remove_file_end_with(ckpt_dir, '*.npz')
                tl.files.save_npz_dict(save_vars, name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

            idx += 1
            global_step += 1

        # test
        idx = 0
        testsummary_loss = 0
        while 1:
            print(idx)
            feed_dict, is_end, is_not_batchsize = data_loader_test.feed_the_network()
            if is_end: break;
            if is_not_batchsize: continue;
            testsummary_loss = sess.run(loss_sum_test_main, feed_dict)
            writer_scalar.add_summary(testsummary_loss, global_step+idx)
            idx += 1

            
        # reset image log
        if epoch % config.TRAIN.refresh_image_log_every == 0:
            writer_image.close()
            remove_file_end_with(log_dir_image, '*.image_log')
            writer_image.reopen()


def evaluate_originalSize():
    print('Evaluation Start')
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/fixed_ckpt'
    save_dir = mode_dir + '/result_video/'

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(save_dir)

    sp_config.batch_size = 1
    batch_size = 1
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.unstab_path_testdataset, regx = '.*', printable = False)))

    for k in np.arange(len(test_video_list)):
        print(test_video_list[k])

    # input
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        if test_video_name != '0.avi' and test_video_name != '1.avi' and test_video_name != '2.avi' and test_video_name != '3.avi' and test_video_name != '6.avi'and test_video_name != '18.avi':
            continue
        print(test_video_name)
        #raise
        tf.reset_default_graph()
        cap = cv2.VideoCapture(config.TEST.unstab_path_testdataset + test_video_name)
        fps = cap.get(5)
        total_frames = cap.get(7)-2
        out_h = int(cap.get(4))
        out_w = int(cap.get(3))
        #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(save_dir + base_name + '_out.avi', fourcc, fps, ( out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            inputs = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')
            resizedInput = tf.placeholder('float32', [batch_size, out_h, out_w , 3], name = 'input_frames')
        # define model
        with tf.variable_scope('main_net') as scope:
            outputs = flownetS_pyramid(inputs,batch_size,is_train=False )
            #outflow = tf.image.resize_images(outputs['predict_flow2'],[out_h,out_w])
            outflow = tf.image.resize_images(outputs['predict_flow2']*384.0/outputs['predict_flow2'].shape.as_list()[1],[out_h,out_w])
            outflow = tf.concat([outflow[:,:,:,0:1]*out_w/512,outflow[:,:,:,1:2]*out_h/384],3)

            #filtersize = 41
            #ofx = outflow[0,:,:,0]
            #for _ in range(filtersize/2):
            #    ofx = tf.pad( ofx, [ [ 1, 1 ], [ 1, 1 ] ], "SYMMETRIC" )
            #ofx = tf.expand_dims(tf.expand_dims(ofx,axis=0),axis=3)
            #smoothofx = tf.nn.conv2d(ofx, tf.constant(1.0/(filtersize*filtersize),shape=[filtersize,filtersize,1,1]),[1,1,1,1], "VALID" )
            #ofy = outflow[0,:,:,1]
            #for _ in range(filtersize/2):
            #    ofy = tf.pad( ofy, [ [ 1, 1 ], [ 1, 1 ] ], "SYMMETRIC" )
            #ofy = tf.expand_dims(tf.expand_dims(ofy,axis=0),axis=3)
            #smoothofy = tf.nn.conv2d(ofy, tf.constant(1.0/(filtersize*filtersize),shape=[filtersize,filtersize,1,1]),[1,1,1,1], "VALID" )
            #smoothoutflow = tf.concat([smoothofx,smoothofy],3)


            outputs_warpedimg = tf_warp(resizedInput, outflow , out_h,out_w)

        # init session
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

        # read frame
        def refine_frame(frame):
            #return cv2.resize(frame / 255., (512, 384))
            return cv2.resize(frame, (512, 384))

        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.resize(frame,(out_w,out_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ref, frame

        
        totaloutputFrame = np.zeros([int(total_frames),out_h,out_w,3])
        #cap.set(1,0)
        #ret_unstab, frame_unstab = cap.read()
        

        for i in range(int(total_frames)):
            print(i)
            print(test_video_name)
            starttime = time.time()
            curinput = np.zeros([1,384,512,27])

            #cap.set(1,i)
            ret_unstab, frame_unstab = cap.read()
            if i == 0:
                totaloutputFrame[0] = frame_unstab
            curinput[0,:,:,24:27] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0  

            #stabidxs = [1,2,3,4,7,15,23,31]
            stabidxs = [31,23,15,7,4,3,2,1]
            for j in range(len(stabidxs)):
                if i - stabidxs[j] < 0:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(cv2.resize(np.uint8(totaloutputFrame[0]), (512, 384)), cv2.COLOR_RGB2BGR))/255.0  
                else:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(cv2.resize(np.uint8(totaloutputFrame[i - stabidxs[j]]), (512, 384)), cv2.COLOR_RGB2BGR))/255.0  

            #feed_dict = {inputs: curinput}
            #curflow = sess.run(outputs , feed_dict)
            #curflow = curflow['predict_flow2']
            #curloc = np.dstack(np.meshgrid(np.arange(out_w), np.arange(out_h)))
            #curloc = curloc + cv2.resize(np.squeeze(curflow),(512,384))
            #curloc = curloc.astype(np.float32)
            #totaloutputFrame[i] = cv2.remap(frame_unstab, curloc, None, cv2.INTER_LINEAR )

            feed_dict = {inputs: curinput, resizedInput: np.expand_dims(cv2.cvtColor(frame_unstab, cv2.COLOR_RGB2BGR)/255.0  ,axis=0)}
            curwarpedimg = sess.run(outputs_warpedimg , feed_dict)

            '''orb = cv2.ORB_create()
            curframe = cv2.cvtColor(np.squeeze(curwarpedimg)*255, cv2.COLOR_RGB2BGR).astype(np.uint8)
            preframe = np.squeeze(totaloutputFrame[np.int32(max(i-1,0))]).astype(np.uint8)
            curframeKeypoint, curframeDescriptors = orb.detectAndCompute(curframe, None)
            prevframeKeypoint, prevframeDescriptors = orb.detectAndCompute(preframe, None)
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(curframeDescriptors, prevframeDescriptors, None)
            matches.sort(key=lambda x: x.distance, reverse=False)
            numGoodMatches = int(len(matches) * 0.6)
            matches = matches[:numGoodMatches]
            validmatchesNum = 0
            validmatchesIdx = []
            for k, match in enumerate(matches):
                is_boundary = False
                for ix in range(-7,7):
                    for iy in range(-7,7):
                        curframelocx = np.int32(min(max(curframeKeypoint[match.queryIdx].pt[1]+ix,0),out_w-1))
                        curframelocy = np.int32(min(max(curframeKeypoint[match.queryIdx].pt[0]+iy,0),out_h-1))
                        prevframelocx = np.int32(min(max(prevframeKeypoint[match.queryIdx].pt[1]+ix,0),out_w-1))
                        prevframelocy = np.int32(min(max(prevframeKeypoint[match.queryIdx].pt[0]+iy,0),out_h-1))
                        if curframe[curframelocy,curframelocx,0] == 0 and curframe[curframelocy,curframelocx,1] == 0 and curframe[curframelocy,curframelocx,2] == 0 \
                            and preframe[prevframelocy,prevframelocx,0] == 0 and preframe[prevframelocy,prevframelocx,1] == 0 and preframe[prevframelocy,prevframelocx,2] == 0 :
                            is_boundary = True
                if not is_boundary:
                    validmatchesNum += 1
                    validmatchesIdx.append(k)

            curframePoints = np.zeros((validmatchesNum, 2), dtype=np.float32)
            prevframePoints = np.zeros((validmatchesNum, 2), dtype=np.float32)
            idx = 0
            for k, match in enumerate(matches):
                if k in validmatchesIdx:
                    curframePoints[idx, :] = curframeKeypoint[match.queryIdx].pt
                    prevframePoints[idx, :] = prevframeKeypoint[match.trainIdx].pt
                    idx += 1
            distanceBtwPoints = np.zeros((validmatchesNum), dtype=np.float32)
            for k in range(validmatchesNum):
                distanceBtwPoints[k] = np.sqrt( (curframePoints[k,0]-prevframePoints[k,0])*(curframePoints[k,0]-prevframePoints[k,0]) + (curframePoints[k,1]-prevframePoints[k,1])*(curframePoints[k,1]-prevframePoints[k,1]) )
            distanceBtwPointsSortedIdx = np.argsort(distanceBtwPoints)
            curframePointsFiltered = curframePoints[distanceBtwPointsSortedIdx[0:max(np.int32(validmatchesNum*0.6),4)],:]
            prevframePointsFiltered = prevframePoints[distanceBtwPointsSortedIdx[0:max(np.int32(validmatchesNum*0.6),4)],:]

            h, mask = cv2.findHomography(curframePointsFiltered, prevframePointsFiltered, cv2.RANSAC)
            curframeHomo = cv2.warpPerspective(curframe, h, (out_w, out_h))
            #h = cv2.getAffineTransform(np.float32(curframePointsFiltered), np.float32(prevframePointsFiltered))
            #curframeHomo= cv2.warpAffine(curframe, h, (out_w, out_h))'''

            '''curinput[0,:,:,24:27] = cv2.resize(cv2.cvtColor(np.squeeze(curwarpedimg), cv2.COLOR_RGB2BGR), (512, 384))
            for j in range(8):
                curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(cv2.resize(np.uint8(totaloutputFrame[max(i-1,0)]), (512, 384)), cv2.COLOR_RGB2BGR))/255.0  
            feed_dict = {inputs: curinput, resizedInput: np.expand_dims(cv2.cvtColor(np.squeeze(curwarpedimg), cv2.COLOR_RGB2BGR),axis=0)}
            curwarpedimgTwice = sess.run(outputs_warpedimg , feed_dict)'''

            #totaloutputFrame[i] = np.squeeze(curwarpedimgTwice)*255
            totaloutputFrame[i] = cv2.cvtColor(np.squeeze(curwarpedimg)*255, cv2.COLOR_RGB2BGR)
            #totaloutputFrame[i] = curframeHomo
            print(time.time()-starttime)
            #out.write(np.concatenate( (preframe, curframe, np.uint8(totaloutputFrame[i])),axis=1))
            #out.write(np.concatenate(( np.uint8(cv2.resize(np.squeeze(curinput[0,:,:,24:27])*255, (out_w, out_h))) ,np.uint8(totaloutputFrame[i])),axis=1))
            out.write(np.uint8(totaloutputFrame[i]))
            
        out.release()    

def evaluate_originalSize_homo():
    print('Evaluation Start')
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/fixed_ckpt'
    save_dir = mode_dir + '/result_video/'

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(save_dir)

    sp_config.batch_size = 1
    batch_size = 1
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.unstab_path_testdataset, regx = '.*', printable = False)))

    for k in np.arange(len(test_video_list)):
        print(test_video_list[k])

    # input
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        
        print(test_video_name)
        if test_video_name != '0.avi' and test_video_name != '1.avi' and test_video_name != '2.avi' and test_video_name != '3.avi' and test_video_name != '6.avi'and test_video_name != '18.avi':
            continue
        #raise
        tf.reset_default_graph()
        cap = cv2.VideoCapture(config.TEST.unstab_path_testdataset + test_video_name)
        fps = cap.get(5)
        total_frames = cap.get(7)-2
        out_h = int(cap.get(4))
        out_w = int(cap.get(3))
        #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(save_dir + base_name + '_out.avi', fourcc, fps, ( out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            inputs = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')
            resizedInput = tf.placeholder('float32', [batch_size, out_h, out_w , 3], name = 'input_frames')
        # define model
        with tf.variable_scope('main_net') as scope:
            outputs = flownetS_pyramid(inputs,batch_size,is_train=False )
            #outflow = tf.image.resize_images(outputs['predict_flow2'],[out_h,out_w])
            outflow = tf.image.resize_images(outputs['predict_flow3']*out_h/outputs['predict_flow3'].shape.as_list()[1],[out_h,out_w])
            outflow = tf.concat([outflow[:,:,:,0:1]*out_w/512,outflow[:,:,:,1:2]*out_h/384],3)
            outputs_warpedimg = tf_warp(resizedInput, outflow , out_h,out_w)

        # init session
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

        # read frame
        def refine_frame(frame):
            #return cv2.resize(frame / 255., (512, 384))
            return cv2.resize(frame, (512, 384))

        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.resize(frame,(out_w,out_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ref, frame

        
        totaloutputFrame = np.zeros([int(total_frames),out_h,out_w,3])
        #cap.set(1,0)
        #ret_unstab, frame_unstab = cap.read()
        

        for i in range(int(total_frames)):
            print(i)
            print(test_video_name)
            starttime = time.time()
            curinput = np.zeros([1,384,512,27])

            #cap.set(1,i)
            ret_unstab, frame_unstab = cap.read()
            if i == 0:
                totaloutputFrame[0] = frame_unstab
            curinput[0,:,:,24:27] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0  

            #stabidxs = [1,2,3,4,7,15,23,31]
            stabidxs = [31,23,15,7,4,3,2,1]
            for j in range(len(stabidxs)):
                if i - stabidxs[j] < 0:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(cv2.resize(np.uint8(totaloutputFrame[0]), (512, 384)), cv2.COLOR_RGB2BGR))/255.0  
                else:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(cv2.resize(np.uint8(totaloutputFrame[i - stabidxs[j]]), (512, 384)), cv2.COLOR_RGB2BGR))/255.0  

            feed_dict = {inputs: curinput, resizedInput: np.expand_dims(cv2.cvtColor(frame_unstab, cv2.COLOR_RGB2BGR)/255.0  ,axis=0)}
            curoutflow = sess.run(outflow , feed_dict)
            curoutputs_warpedimg = sess.run(outputs_warpedimg , feed_dict)
            curoutflow = np.squeeze(curoutflow)
            #curoutflow = np.concatenate((curoutflow[:,:,1:2],curoutflow[:,:,0:1]),axis=2)

            xv,yv = np.meshgrid(np.linspace(0,out_w-1,out_w),np.linspace(0,out_h-1,out_h))
            gridmesh = np.concatenate((np.expand_dims(xv,2),np.expand_dims(yv,2)),axis=2)
            gridmeshOF = gridmesh - curoutflow

            #print(gridmesh.shape)
            #print(np.reshape(gridmesh,(out_w*out_h*2)).shape)
            #h, mask = cv2.findHomography(np.reshape(gridmesh,(out_w*out_h,2)), np.reshape(gridmeshOF,(out_w*out_h,2)), cv2.RANSAC)
            h, mask = cv2.findHomography(np.reshape(gridmesh,(out_w*out_h,2)), np.reshape(gridmeshOF,(out_w*out_h,2)), cv2.RANSAC)
            curframeHomo = cv2.warpPerspective(frame_unstab, h, (out_w, out_h))

            #totaloutputFrame[i] = np.squeeze(curwarpedimgTwice)*255
            #totaloutputFrame[i] = curframeHomo
            totaloutputFrame[i] = cv2.cvtColor(np.squeeze(curoutputs_warpedimg)*255, cv2.COLOR_RGB2BGR)


            print(time.time()-starttime)
            out.write(np.uint8(curframeHomo))
            #out.write(np.uint8(totaloutputFrame[i]))

            #out.write(np.concatenate( (np.uint8(np.squeeze(curoutputs_warpedimg*255)), np.uint8(totaloutputFrame[i])),axis=1))

        out.release()    

def evaluate():
    print('Evaluation Start')
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/fixed_ckpt'
    save_dir = mode_dir + '/result_video/'

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(save_dir)

    sp_config.batch_size = 1
    batch_size = 1
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.unstab_path_testdataset, regx = '.*', printable = False)))

    for k in np.arange(len(test_video_list)):
        print(test_video_list[k])

    # input
    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        #if test_video_name == '18.avi' or test_video_name == '18AF.avi' or test_video_name == '6.avi':
        #if test_video_name != '18.avi' and  test_video_name != '18AF.avi' and test_video_name != '6.avi' and  test_video_name != '0.avi': # and test_video_name != '3.avi':
        #if test_video_name != '3.avi' and test_video_name != '6.avi' and test_video_name != '652.avi' and test_video_name != '653.avi' and test_video_name != '654.avi':
        #if test_video_name != '100.avi' and  test_video_name != '101.avi' and test_video_name != '102.avi' and  test_video_name != '103.avi': # and test_video_name != '3.avi':
        #    continue
        
        print(test_video_name)
        #raise
        tf.reset_default_graph()
        cap = cv2.VideoCapture(config.TEST.unstab_path_testdataset + test_video_name)
        fps = cap.get(5)
        total_frames = cap.get(7)-2
        out_h = 384 #int(cap.get(4))
        out_w = 512 #int(cap.get(3))
        #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(save_dir + base_name + '_out.avi', fourcc, fps, (2 * out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            inputs = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')
        # define model
        with tf.variable_scope('main_net') as scope:
            outputs = flownetS_pyramid(inputs,batch_size,is_train=False )
            unstabimg = tf.image.resize_images(inputs[:,:,:,24:27],[382,510])
            outputs_warpedimg = tf_warp(unstabimg, outputs['predict_flow2'], 382, 510)

        # init session
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

        # read frame
        def refine_frame(frame):
            #return cv2.resize(frame / 255., (512, 384))
            return cv2.resize(frame, (512, 384))

        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.resize(frame,(out_w,out_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ref, frame

        
        totaloutputFrame = np.zeros([int(total_frames),out_h,out_w,3])
        #cap.set(1,0)
        #ret_unstab, frame_unstab = cap.read()
        

        for i in range(int(total_frames)):
            print(i)
            starttime = time.time()
            curinput = np.zeros([1,out_h,out_w,27])

            #cap.set(1,i)
            ret_unstab, frame_unstab = cap.read()
            if i == 0:
                totaloutputFrame[0] = cv2.resize(frame_unstab, (512, 384))
            curinput[0,:,:,24:27] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0  

            stabidxs = [1,2,3,4,7,15,23,31]
            #stabidxs = [31,23,15,7,4,3,2,1]
            for j in range(len(stabidxs)):
                if i - stabidxs[j] < 0:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(np.uint8(totaloutputFrame[0]), cv2.COLOR_RGB2BGR))/255.0  
                else:
                    curinput[0,:,:,j*3:(j+1)*3] = np.float32(cv2.cvtColor(np.uint8(totaloutputFrame[i - stabidxs[j]]), cv2.COLOR_RGB2BGR))/255.0  

            #feed_dict = {inputs: curinput}
            #curflow = sess.run(outputs , feed_dict)
            #curflow = curflow['predict_flow2']
            #curloc = np.dstack(np.meshgrid(np.arange(out_w), np.arange(out_h)))
            #curloc = curloc + cv2.resize(np.squeeze(curflow),(512,384))
            #curloc = curloc.astype(np.float32)
            #totaloutputFrame[i] = cv2.remap(frame_unstab, curloc, None, cv2.INTER_LINEAR )

            feed_dict = {inputs: curinput}
            curwarpedimg = sess.run(outputs_warpedimg , feed_dict)
            totaloutputFrame[i] = cv2.cvtColor(cv2.resize(np.squeeze(curwarpedimg),(512,384))*255, cv2.COLOR_RGB2BGR)
            print(time.time()-starttime)
            out.write(np.uint8( np.concatenate([np.uint8(cv2.resize(frame_unstab, (512, 384))), np.uint8(totaloutputFrame[i]) ],axis=1)))
            
        out.release()    

# imported from flownet
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    anglemap = np.expand_dims(np.uint8(np.arctan2(-v, -u) / np.pi * 127+ 128),axis=2)
    anglemap = np.concatenate([anglemap,anglemap,anglemap],axis=2)

    return np.uint8(img), anglemap
    
# imported from flownet
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default = 'sharp_ass', help = 'model name')
    parser.add_argument('--is_train', type = str , default = 'true', help = 'whether to train or not')
    parser.add_argument('--delete_log', type = str , default = 'false', help = 'whether to delete log or not')
    parser.add_argument('--is_acc', type = str , default = 'false', help = 'whether to train or not')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['is_train'] = t_or_f(args.is_train)
    tl.global_flag['delete_log'] = t_or_f(args.delete_log)
    
    tl.global_flag['is_acc'] = t_or_f(args.is_acc)
    tl.logging.set_verbosity(tl.logging.INFO)

    if tl.global_flag['is_train']:
        train()
    elif tl.global_flag['is_acc']:
        get_accuracy()
    else:
        evaluate_originalSize()
        #evaluate_video_interpolation_avgof()

