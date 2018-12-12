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
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
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
    
    ## DEFINE MODEL
    # input
    with tf.variable_scope('input'):
        #patches_in = tf.placeholder('float32', [batch_size, sample_num, h, w, 3], name = 'input_frames')
        unstab_image = tf.placeholder('float32', [batch_size, 384, 512, 3], name = 'input_frames')
        stab_image = tf.placeholder('float32', [batch_size, 384, 512, 3*8], name = 'input_flows')

    with tf.variable_scope('main_net') as scope:
        outputs = flownetS_pyramid( tf.concat([stab_image,unstab_image],3) ,batch_size,is_train=True )
        outputs_test = flownetS_pyramid(tf.concat([stab_image,unstab_image],3),batch_size,is_train=False,reuse=True )


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
        return masked_MSE(warpedimg, downsampled_stab, mask),warpedimg


    ## DEFINE LOSS
    with tf.variable_scope('loss'):

        loss6,warpedimg6 = lossterm(outputs['predict_flow6'],stab_image[:,:,:,0:3],unstab_image)
        loss5,warpedimg5 = lossterm(outputs['predict_flow5'],stab_image[:,:,:,0:3],unstab_image)
        loss4,warpedimg4 = lossterm(outputs['predict_flow4'],stab_image[:,:,:,0:3],unstab_image)
        loss3,warpedimg3 = lossterm(outputs['predict_flow3'],stab_image[:,:,:,0:3],unstab_image)
        loss2,warpedimg2 = lossterm(outputs['predict_flow2'],stab_image[:,:,:,0:3],unstab_image)

        loss6_1,_ = lossterm(outputs['predict_flow6'],stab_image[:,:,:,3:6],unstab_image)
        loss5_1,_ = lossterm(outputs['predict_flow5'],stab_image[:,:,:,3:6],unstab_image)
        loss4_1,_ = lossterm(outputs['predict_flow4'],stab_image[:,:,:,3:6],unstab_image)
        loss3_1,_ = lossterm(outputs['predict_flow3'],stab_image[:,:,:,3:6],unstab_image)
        loss2_1,_ = lossterm(outputs['predict_flow2'],stab_image[:,:,:,3:6],unstab_image)
        loss_1 = (loss6_1+loss5_1+loss4_1+loss3_1+loss2_1)*np.exp(-1/3)

        loss6_2,_ = lossterm(outputs['predict_flow6'],stab_image[:,:,:,6:9],unstab_image)
        loss5_2,_ = lossterm(outputs['predict_flow5'],stab_image[:,:,:,6:9],unstab_image)
        loss4_2,_ = lossterm(outputs['predict_flow4'],stab_image[:,:,:,6:9],unstab_image)
        loss3_2,_ = lossterm(outputs['predict_flow3'],stab_image[:,:,:,6:9],unstab_image)
        loss2_2,_ = lossterm(outputs['predict_flow2'],stab_image[:,:,:,6:9],unstab_image)
        loss_2 = (loss6_2+loss5_2+loss4_2+loss3_2+loss2_2)*np.exp(-2/3)

        loss6_3,_ = lossterm(outputs['predict_flow6'],stab_image[:,:,:,9:12],unstab_image)
        loss5_3,_ = lossterm(outputs['predict_flow5'],stab_image[:,:,:,9:12],unstab_image)
        loss4_3,_ = lossterm(outputs['predict_flow4'],stab_image[:,:,:,9:12],unstab_image)
        loss3_3,_ = lossterm(outputs['predict_flow3'],stab_image[:,:,:,9:12],unstab_image)
        loss2_3,_ = lossterm(outputs['predict_flow2'],stab_image[:,:,:,9:12],unstab_image)
        loss_3 = (loss6_3+loss5_3+loss4_3+loss3_3+loss2_3)*np.exp(-3/3)

        loss6_4,_ = lossterm(outputs['predict_flow6'],stab_image[:,:,:,12:15],unstab_image)
        loss5_4,_ = lossterm(outputs['predict_flow5'],stab_image[:,:,:,12:15],unstab_image)
        loss4_4,_ = lossterm(outputs['predict_flow4'],stab_image[:,:,:,12:15],unstab_image)
        loss3_4,_ = lossterm(outputs['predict_flow3'],stab_image[:,:,:,12:15],unstab_image)
        loss2_4,_ = lossterm(outputs['predict_flow2'],stab_image[:,:,:,12:15],unstab_image)
        loss_4 = (loss6_4+loss5_4+loss4_4+loss3_4+loss2_4)*np.exp(-6/3)

        loss6_5,_ = lossterm(outputs['predict_flow6'],stab_image[:,:,:,15:18],unstab_image)
        loss5_5,_ = lossterm(outputs['predict_flow5'],stab_image[:,:,:,15:18],unstab_image)
        loss4_5,_ = lossterm(outputs['predict_flow4'],stab_image[:,:,:,15:18],unstab_image)
        loss3_5,_ = lossterm(outputs['predict_flow3'],stab_image[:,:,:,15:18],unstab_image)
        loss2_5,_ = lossterm(outputs['predict_flow2'],stab_image[:,:,:,15:18],unstab_image)
        loss_5 = (loss6_5+loss5_5+loss4_5+loss3_5+loss2_5)*np.exp(-14/3)
        
        var6 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow6']))/2e7
        var5 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow5']))/2e7
        var4 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow4']))/2e7
        var3 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow3']))/2e7
        var2 = tf.reduce_sum(tf.image.total_variation(outputs['predict_flow2']))/2e7

        loss_main = tf.identity( loss6+loss5+loss4+loss3+loss2+var6+var5+var3+var2+loss_1+loss_2+loss_3+loss_4+loss_5, name = 'total')
            
    with tf.variable_scope('loss_test'):

        loss6,warpedimg6 = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,0:3],unstab_image)
        loss5,warpedimg5 = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,0:3],unstab_image)
        loss4,warpedimg4 = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,0:3],unstab_image)
        loss3,warpedimg3 = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,0:3],unstab_image)
        loss2,warpedimg2 = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,0:3],unstab_image)

        loss6_1,_ = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,3:6],unstab_image)
        loss5_1,_ = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,3:6],unstab_image)
        loss4_1,_ = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,3:6],unstab_image)
        loss3_1,_ = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,3:6],unstab_image)
        loss2_1,_ = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,3:6],unstab_image)
        loss_1 = (loss6_1+loss5_1+loss4_1+loss3_1+loss2_1)*np.exp(-1/3)

        loss6_2,_ = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,6:9],unstab_image)
        loss5_2,_ = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,6:9],unstab_image)
        loss4_2,_ = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,6:9],unstab_image)
        loss3_2,_ = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,6:9],unstab_image)
        loss2_2,_ = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,6:9],unstab_image)
        loss_2 = (loss6_2+loss5_2+loss4_2+loss3_2+loss2_2)*np.exp(-2/3)

        loss6_3,_ = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,9:12],unstab_image)
        loss5_3,_ = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,9:12],unstab_image)
        loss4_3,_ = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,9:12],unstab_image)
        loss3_3,_ = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,9:12],unstab_image)
        loss2_3,_ = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,9:12],unstab_image)
        loss_3 = (loss6_3+loss5_3+loss4_3+loss3_3+loss2_3)*np.exp(-3/3)

        loss6_4,_ = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,12:15],unstab_image)
        loss5_4,_ = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,12:15],unstab_image)
        loss4_4,_ = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,12:15],unstab_image)
        loss3_4,_ = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,12:15],unstab_image)
        loss2_4,_ = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,12:15],unstab_image)
        loss_4 = (loss6_4+loss5_4+loss4_4+loss3_4+loss2_4)*np.exp(-6/3)

        loss6_5,_ = lossterm(outputs_test['predict_flow6'],stab_image[:,:,:,15:18],unstab_image)
        loss5_5,_ = lossterm(outputs_test['predict_flow5'],stab_image[:,:,:,15:18],unstab_image)
        loss4_5,_ = lossterm(outputs_test['predict_flow4'],stab_image[:,:,:,15:18],unstab_image)
        loss3_5,_ = lossterm(outputs_test['predict_flow3'],stab_image[:,:,:,15:18],unstab_image)
        loss2_5,_ = lossterm(outputs_test['predict_flow2'],stab_image[:,:,:,15:18],unstab_image)
        loss_5 = (loss6_5+loss5_5+loss4_5+loss3_5+loss2_5)*np.exp(-14/3)


        var6 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow6']))/2e7
        var5 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow5']))/2e7
        var4 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow4']))/2e7
        var3 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow3']))/2e7
        var2 = tf.reduce_sum(tf.image.total_variation(outputs_test['predict_flow2']))/2e7

        loss_main_test = tf.identity( loss6+loss5+loss4+loss3+loss2+var6+var5+var3+var2+loss_1+loss_2+loss_3+loss_4+loss_5   , name = 'totaltest')
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
    image_sum_list.append(tf.summary.image('unstab_image', fix_image_tf(unstab_image, 1)))
    image_sum_list.append(tf.summary.image('stab_image', fix_image_tf(stab_image[:,:,:,0:3], 1)))
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
    #tl.files.load_and_assign_npz_dict(name = '/Jarvis/logs/LJH/deep_video_stabilization/VS_flownetS_realdata/checkpoint_load/VS_flownetS_realdata.npz', sess = sess)

    ## START TRAINING
    print '*****************************************'
    print '             TRAINING START'
    print '*****************************************'
    global_step = 0
    for epoch in range(0, n_epoch + 1):
        total_loss, n_iter = 0, 0

        stab_video_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.stab_path, regx = '.*', printable = False)))
        unstab_video_list = np.array(sorted(tl.files.load_file_list(path = config.TRAIN.unstab_path, regx = '.*', printable = False)))
        test_stab_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.stab_path, regx = '.*', printable = False)))
        test_unstab_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.unstab_path, regx = '.*', printable = False)))

        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(learning_rate, lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(learning_rate, lr_init))

        epoch_time = time.time()
        for frame_idx in range(0, 10000000):
            print('==============================================================')
            step_time = time.time()
            curUnstabImg = np.zeros([batch_size, 384, 512, 3])
            curStabImg = np.zeros([batch_size, 384, 512, 3*8])

            for i in range(batch_size):
                randCapidx = random.randrange(0,len(stab_video_list))
                cap_stab = cv2.VideoCapture(config.TRAIN.stab_path + stab_video_list[randCapidx])
                cap_unstab = cv2.VideoCapture(config.TRAIN.unstab_path + unstab_video_list[randCapidx])
                total_frames = int(cap_stab.get(7))
                randFrameidx = random.randrange(1,total_frames)

                cap_unstab.set(1,randFrameidx)
                ret_unstab, frame_unstab = cap_unstab.read()
                curUnstabImg[i] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0
                
                #cap_stab.set(1,randFrameidx)
                #ret_stab, frame_stab = cap_stab.read()
                stabidxs = [1,2,3,4,7,15,23,31]
                for j in range(len(stabidxs)):
                    if randFrameidx - stabidxs[j] < 0:
                        cap_stab.set(1,0)
                    else:
                        cap_stab.set(1,randFrameidx - stabidxs[j])
                    ret_stab, frame_stab = cap_stab.read()
                    curStabImg[i,:,:,j*3:(j+1)*3] = cv2.cvtColor(cv2.resize(frame_stab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0
                

            ## RUN NETWORK
            feed_dict = {unstab_image:curUnstabImg, stab_image:curStabImg}
            _ = sess.run(optim_main, feed_dict)

            # print status
            err_main, lr = \
            sess.run([loss_main, learning_rate], feed_dict)
            print('[%s]  %4d/%4d time: %4.2fs, err[main: %1.2e], lr: %1.2e' % \
                (tl.global_flag['mode'], frame_idx, 10000000, time.time() - step_time, err_main, lr))
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
            # save test loss 
            if  global_step % config.TRAIN.write_ckpt_every == 0:
                testcurUnstabImg = np.zeros([batch_size, 384, 512, 3])
                testcurStabImg = np.zeros([batch_size, 384, 512, 3*8])
                for k in range(10):
                    for i in range(batch_size):
                        randCapidx = random.randrange(0,len(test_stab_video_list))
                        cap_stab = cv2.VideoCapture(config.TEST.stab_path + test_stab_video_list[randCapidx])
                        cap_unstab = cv2.VideoCapture(config.TEST.unstab_path + test_unstab_video_list[randCapidx])
                        total_frames = int(cap_stab.get(7))
                        randFrameidx = random.randrange(1,total_frames)

                        cap_unstab.set(1,randFrameidx)
                        ret_unstab, frame_unstab = cap_unstab.read()
                        #print(ret_unstab)
                        testcurUnstabImg[i] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0
                        
                        #cap_stab.set(1,randFrameidx)
                        #ret_stab, frame_stab = cap_stab.read()
                        stabidxs = [1,2,3,4,7,15,23,31]
                        for j in range(len(stabidxs)):
                            if randFrameidx - stabidxs[j] < 0:
                                cap_stab.set(1,0)
                            else:
                                cap_stab.set(1,randFrameidx - stabidxs[j])
                            ret_stab, frame_stab = cap_stab.read()
                            testcurStabImg[i,:,:,j*3:(j+1)*3] = cv2.cvtColor(cv2.resize(frame_stab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0

                    feed_dict = {unstab_image:testcurUnstabImg, stab_image:testcurStabImg}  
                    testsummary_loss  = sess.run(loss_sum_test_main, feed_dict)
                    writer_scalar.add_summary(testsummary_loss, global_step+k-1)

            global_step += 1

            
        #print('[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %1.2e' % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter/n_frames))
        # reset image log
        if epoch % config.TRAIN.refresh_image_log_every == 0:
            writer_image.close()
            remove_file_end_with(log_dir_image, '*.image_log')
            writer_image.reopen()


def evaluate():
    print 'Evaluation Start'
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


def evaluate_blurNma():
    print 'Evaluation Start'
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
        if test_video_name != '18.avi' and  test_video_name != '18AF.avi' and test_video_name != '6.avi' and  test_video_name != '0.avi': # and test_video_name != '3.avi':
        #if test_video_name != '3.avi' and test_video_name != '6.avi' and test_video_name != '652.avi' and test_video_name != '653.avi' and test_video_name != '654.avi':
        #if test_video_name != '100.avi' and  test_video_name != '101.avi' and test_video_name != '102.avi' and  test_video_name != '103.avi': # and test_video_name != '3.avi':
            continue
        
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

        out = cv2.VideoWriter(save_dir + base_name + '_out_blurNma.avi', fourcc, fps, (2 * out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            inputs = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')
            inputs_prevof = tf.placeholder('float32', [batch_size, 382, 510, 2], name = 'input_prevof')
        # define model
        with tf.variable_scope('main_net') as scope:
            outputs = flownetS_pyramid(inputs,batch_size,is_train=False )
            unstabimg = tf.image.resize_images(inputs[:,:,:,24:27],[382,510])
            of = outputs['predict_flow2']
            
            ofx = tf.expand_dims(of[:,:,:,0],axis=3)
            smoothofx = tf.nn.conv2d(ofx, tf.constant(1/(75*75.0),shape=[75,75,1,1]),[1,1,1,1], "SAME" )
            ofy = tf.expand_dims(of[:,:,:,1],axis=3)
            smoothofy = tf.nn.conv2d(ofy, tf.constant(1/(75*75.0),shape=[75,75,1,1]),[1,1,1,1], "SAME" )
            smoothof = tf.concat([smoothofx,smoothofy],3)
            print(smoothof)
            
            outputs_warpedimg = tf_warp(unstabimg, 0.9*smoothof+0.1*inputs_prevof , 382, 510)

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
        prevof = np.zeros([382,510,2]);

        for i in range(int(total_frames)):
            print(i)
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

            feed_dict = {inputs: curinput, inputs_prevof: np.expand_dims(prevof,axis=0)}
            curwarpedimg,curof = sess.run((outputs_warpedimg,of) , feed_dict)
            prevof = 0.9*prevof + 0.1*np.squeeze(curof)
            totaloutputFrame[i] = cv2.cvtColor(cv2.resize(np.squeeze(curwarpedimg),(512,384))*255, cv2.COLOR_RGB2BGR)

            out.write(np.uint8( np.concatenate([np.uint8(cv2.resize(frame_unstab, (512, 384))), np.uint8(totaloutputFrame[i]) ],axis=1)))
            
        out.release()    


def evaluate_medianNma():
    print 'Evaluation Start'
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
        if test_video_name != '18.avi' and  test_video_name != '18AF.avi' and test_video_name != '6.avi' and  test_video_name != '0.avi': # and test_video_name != '3.avi':
        #if test_video_name != '3.avi' and test_video_name != '6.avi' and test_video_name != '652.avi' and test_video_name != '653.avi' and test_video_name != '654.avi':
        #if test_video_name != '100.avi' and  test_video_name != '101.avi' and test_video_name != '102.avi' and  test_video_name != '103.avi': # and test_video_name != '3.avi':
            continue
        
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

        out = cv2.VideoWriter(save_dir + base_name + '_out_mdeianNma.avi', fourcc, fps, (2 * out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            inputs = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')
        # define model
        with tf.variable_scope('main_net') as scope:
            outputs = flownetS_pyramid(inputs,batch_size,is_train=False )
            unstabimg = tf.image.resize_images(inputs[:,:,:,24:27],[382,510])
            of = outputs['predict_flow2']
        

        with tf.variable_scope('input_warp'):
            inputs2 = tf.placeholder('float32', [batch_size, 384, 512, 3*9], name = 'input_frames')   
            inputs_prevof = tf.placeholder('float32', [batch_size, 382, 510, 2], name = 'input_prevof')
            medianof = tf.placeholder('float32', [batch_size, 382, 510, 2], name = 'input_medianof')
        with tf.variable_scope('main_net_warp') as scope:
            unstabimg = tf.image.resize_images(inputs2[:,:,:,24:27],[382,510])
            outputs_warpedimg = tf_warp(unstabimg, 0.9*medianof+0.1*inputs_prevof , 382, 510)

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
        prevof = np.zeros([382,510,2]);

        for i in range(int(total_frames)):
            print(i)
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

            feed_dict = {inputs: curinput}
            curof = sess.run(of , feed_dict)
            #curmedianof = np.concatenate( ( np.expand_dims(cv2.medianBlur(curof[:,:,0], 101),2),np.expand_dims(cv2.medianBlur(curof[:,:,1], 101),2) ) , axis=2)
            #curof = scipy.ndimage.filters.median_filter(curof,31)
            curmedianof = scipy.signal.medfilt(np.squeeze(curof),5)
            print(curmedianof.shape)

            feed_dict = {inputs: curinput, inputs_prevof: np.expand_dims(prevof,axis=0),medianof: curmedianof}
            curwarpedimg = sess.run(outputs_warpedimg , feed_dict)

            prevof = 0.9*prevof + 0.1*np.squeeze(curmedianof)

            totaloutputFrame[i] = cv2.cvtColor(cv2.resize(np.squeeze(curwarpedimg),(512,384))*255, cv2.COLOR_RGB2BGR)

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
        evaluate()
        #evaluate_video_interpolation_avgof()

