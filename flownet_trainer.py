from config import sp_config, config, log_config
from utils import *
from model import *
from spatial_transformer import ProjectiveSymmetryTransformer, ProjectiveTransformer, AffineSymmetryTransformer,SimilarityTransformer
from tensorlayer.layers import *
import NLDF
from cell import ConvLSTMCell, ConvGRUCell
import os

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
from enum import Enum
import os
import uuid
from flownet.src.net import Mode
from flownet.src.flownet_s.flownet_s import FlowNetS
from flownet.src.flownet_sd.flownet_sd import FlowNetSD
from flownet.src.training_schedules import LONG_SCHEDULE

slim = tf.contrib.slim
flownetSD = FlowNetSD(mode=Mode.TRAIN)
flownetS = FlowNetS(mode=Mode.TRAIN)

batch_size = config.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

h = config.height
w = config.width

ni = int(np.ceil(np.sqrt(batch_size)))

sample_num = config.seq_length
source_idx = sample_num - 1

prev_output = np.zeros([config.prev_output_num,config.height,config.width,3])


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
        patches_stab = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_frames_stab')
        patches_unstab = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_frames_unstab')

    inputs = {
        'input_a': tf.image.resize_bilinear(patch_unstab,tf.constant([192,256])),
        'input_b': tf.image.resize_bilinear(patch_stab,tf.constant([192,256]))
    }
    training_schedule = LONG_SCHEDULE
    predictions = flownetS.model(inputs, training_schedule)
    pred_flow = predictions['flow']
    patch_warped = tf_warp(tf.image.resize_bilinear(patch_unstab,tf.constant([192,256])),-1*pred_flow,192,256)
   
    ## define loss

    def masked_MSE(pred, gt, mask):
        pred_mask = pred * mask
        gt_mask = gt * mask
        MSE = tf.reduce_sum(tf.squared_difference(pred_mask, gt_mask), axis = [1, 2, 3])
        safe_mask = tf.cast(tf.where(tf.equal(mask, tf.zeros_like(mask)), mask + tf.constant(1e-8), mask), tf.float32)
        MSE = MSE / tf.reduce_sum(safe_mask, axis = [1, 2, 3])
        return tf.reduce_mean(MSE)
    def lossterm(predict_flow,stab_image,unstab_image):
        size = [predict_flow.shape[1], predict_flow.shape[2]]
        downsampled_stab = tf.image.resize_images(stab_image, size )#downsample(gt_flow, size)
        downsampled_unstab = tf.image.resize_images(unstab_image, size )#downsample(gt_flow, size)
        warpedimg = tf_warp(downsampled_unstab, predict_flow, predict_flow.shape[1], predict_flow.shape[2])
        #loss6 = tl.cost.mean_squared_error(downsampled_stab6, warpedimg6, is_mean = True, name = 'loss6') #* 0.32
        mask = tf_warp(tf.ones_like(downsampled_unstab), predict_flow, predict_flow.shape[1], predict_flow.shape[2])
        return masked_MSE(warpedimg, downsampled_stab, mask),warpedimg

    with tf.variable_scope('loss'):
        loss6,warpedimg6 = lossterm(-20 * predictions['predict_flow6'],patches_stab,patches_unstab)
        loss5,warpedimg5 = lossterm(-20 * predictions['predict_flow5'],patches_stab,patches_unstab)
        loss4,warpedimg4 = lossterm(-20 * predictions['predict_flow4'],patches_stab,patches_unstab)
        loss3,warpedimg3 = lossterm(-20 * predictions['predict_flow3'],patches_stab,patches_unstab)
        loss2,warpedimg2 = lossterm(-20 * predictions['predict_flow2'],patches_stab,patches_unstab)
        loss1,warpedimg1 = lossterm(-1 * predictions['flow'],patches_stab,patches_unstab)


    loss_main = tf.identity( loss6+loss5+loss4+loss3+loss2+loss1, name = 'total')

    ## DEFINE OPTIMIZER
    # variables to save / train
    main_vars = tl.layers.get_variables_with_name('FlowNetS', True, False)
    save_vars = tl.layers.get_variables_with_name('FlowNetS', False, False)

    # define optimizer
    with tf.variable_scope('Optimizer'):
        learning_rate = tf.Variable(lr_init, trainable = False)
        optim_main = tf.train.AdamOptimizer(learning_rate, beta1 = beta1).minimize(loss_main, var_list = main_vars)


    ## DEFINE SUMMARY
    # writer
    writer_scalar = tf.summary.FileWriter(log_dir_scalar, sess.graph, flush_secs=30, filename_suffix = '.loss_log')
    writer_image = tf.summary.FileWriter(log_dir_image, sess.graph, flush_secs=30, filename_suffix = '.image_log')

    loss_sum_main_list = []
    with tf.variable_scope('loss'):
        loss_sum_main_list.append(tf.summary.scalar('loss6', loss6))
        loss_sum_main_list.append(tf.summary.scalar('loss5', loss5))
        loss_sum_main_list.append(tf.summary.scalar('loss4', loss4))
        loss_sum_main_list.append(tf.summary.scalar('loss3', loss3))
        loss_sum_main_list.append(tf.summary.scalar('loss2', loss2))
        loss_sum_main_list.append(tf.summary.scalar('loss1', loss1))
        loss_sum_main_list.append(tf.summary.scalar('loss_main', loss_main))

    loss_sum_main = tf.summary.merge(loss_sum_main_list)

    image_sum_list = []
    image_sum_list.append(tf.summary.image('patch_unstab_source', fix_image_tf(patch_unstab, 1)))
    image_sum_list.append(tf.summary.image('patch_stab_source', fix_image_tf(patch_stab, 1)))
    image_sum_list.append(tf.summary.image('patch_warped', fix_image_tf(patch_warped, 1)))
    image_sum_list.append(tf.summary.image('warpedimg6', fix_image_tf(warpedimg6, 1)))
    image_sum_list.append(tf.summary.image('warpedimg5', fix_image_tf(warpedimg5, 1)))
    image_sum_list.append(tf.summary.image('warpedimg4', fix_image_tf(warpedimg4, 1)))
    image_sum_list.append(tf.summary.image('warpedimg3', fix_image_tf(warpedimg3, 1)))
    image_sum_list.append(tf.summary.image('warpedimg2', fix_image_tf(warpedimg2, 1)))

    image_sum = tf.summary.merge(image_sum_list)

    ## INITIALIZE SESSION
    tl.layers.initialize_global_variables(sess)
    flownetSaver = tf.train.Saver(tl.layers.get_variables_with_name('FlowNetS'))
    flownetSaver.restore(sess, './flownet/checkpoints/FlowNetS/flownet-S.ckpt-0')
    #tl.files.load_and_assign_npz_dict(name = '/Jarvis/logs/LJH/deep_video_stabilization/VS_addhomo_Limit_checkpoint/checkpoint/VS_addhomo_Limit.npz', sess = sess)

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
            curStabImg = np.zeros([batch_size, 384, 512, 3])

            for i in range(batch_size):
                randCapidx = random.randrange(0,len(stab_video_list))
                cap_stab = cv2.VideoCapture(config.TRAIN.stab_path + stab_video_list[randCapidx])
                cap_unstab = cv2.VideoCapture(config.TRAIN.unstab_path + unstab_video_list[randCapidx])
                total_frames = int(cap_stab.get(7))
                randFrameidx = random.randrange(1,total_frames)

                cap_unstab.set(1,randFrameidx)
                ret_unstab, frame_unstab = cap_unstab.read()
                curUnstabImg[i] = cv2.cvtColor(cv2.resize(frame_unstab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0
                
                cap_stab.set(1,randFrameidx)
                ret_stab, frame_stab = cap_stab.read()
                curStabImg[i] = cv2.cvtColor(cv2.resize(frame_stab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0
                    
                

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
            """if  global_step % config.TRAIN.write_ckpt_every == 0:
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
                        
                        cap_stab.set(1,randFrameidx)
                        ret_stab, frame_stab = cap_stab.read()
                        testcurStabImg[i] = cv2.cvtColor(cv2.resize(frame_stab, (512, 384)), cv2.COLOR_RGB2BGR)/255.0

                    feed_dict = {unstab_image:testcurUnstabImg, stab_image:testcurStabImg}  
                    testsummary_loss  = sess.run(loss_sum_test_main, feed_dict)
                    writer_scalar.add_summary(testsummary_loss, global_step+k-1)"""

            global_step += 1

            
        #print('[*] Epoch: [%2d/%2d] time: %4.4fs, total_err: %1.2e' % (epoch, n_epoch, time.time() - epoch_time, total_loss/n_iter/n_frames))
        # reset image log
        if epoch % config.TRAIN.refresh_image_log_every == 0:
            writer_image.close()
            remove_file_end_with(log_dir_image, '*.image_log')
            writer_image.reopen()

"""def evaluate():
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/fixed_ckpt'
    save_dir = mode_dir + '/result/'

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(save_dir)

    sp_config.batch_size = 1
    
    # input
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.dataset_path, regx = '.*', printable = False)))

    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        tf.reset_default_graph()
        cap = cv2.VideoCapture(config.TEST.dataset_path + test_video_name)
        fps = cap.get(5)
        h = int(cap.get(4))
        w = int(cap.get(3))
        refine_temp = np.ones((h, w))
        refine_temp = refine_image(refine_temp)
        [h, w] = refine_temp.shape[:2]
        total_frames = cap.get(7)
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        #out = cv2.VideoWriter('output.avi',fourcc, fps, (h, w))
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(save_dir + base_name + '_out.avi', fourcc, fps, (2 * config.width,config.height))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            patches_prev_output = tf.placeholder('float32', [batch_size, config.prev_output_num, h, w, 3], name = 'input_frames_stab')
            patches_unstab = tf.placeholder('float32', [batch_size, sample_num, h, w, 3], name = 'input_frames_unstab')
            patch_unstab_source = patches_unstab[:, source_idx, :, :, :]
            patch_unstab_source = tf.reshape(patch_unstab_source, [batch_size, h, w, 3])

        # define model
        with tf.variable_scope('main_net') as scope:
            with tf.variable_scope('stab_net') as scope:
                with tf.variable_scope('2Dconv') as scope:
                    _,_,_,_,conv2dOut = UNet_down(patch_unstab_source, config.num_features, is_train=True, reuse = False, scope = scope)
                with tf.variable_scope('3Dconv') as scope:
                    conv3dOut = UNet_down_3D(patches_prev_output, is_train=True, reuse = False, scope = scope)
                with tf.variable_scope('merge') as scope:
                    convMerged = tf.concat([conv2dOut,conv3dOut],3)
                    print(convMerged)
                    convMerged = UNet_merged(convMerged, is_train=True, reuse=False, scope = scope)
                    homoTheta = dense_homo(convMerged,is_train=False, reuse=False, scope = scope);
                    #unetUpWOu4 = UNet_up_merged_without_u4(convMerged, is_train=False, reuse=False, scope = scope)

                    stl_projective = ProjectiveTransformer(sp_config.out_size)
                    HomoOutput = stl_projective.transform(patch_unstab_source, homoTheta )
                    output = HomoOutput # + unetUpWOu4

        # init session
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

        frames_unstab = []
        for frame_idx in range(int(total_frames)):
            frames_unstab, train_frames_unstab, is_end_unstab = get_frame_batch(frames_unstab, cap, sample_num = sample_num, batch_size = batch_size, n_frames = n_frames, resize_shape = [w, h], upper_limit_to_resize = 300)
            train_frames_unstab= resize_frames( train_frames_unstab, [w, h], batch_size, sample_num)

            if frame_idx==0:   
                for i in range(config.prev_output_num):
                    prev_output[i] = train_frames_unstab[0, source_idx, :, :, :]

            feed_dict = {patches_unstab: train_frames_unstab,patches_prev_output: np.expand_dims(prev_output,axis=0)}
            curoutput  = sess.run(output, feed_dict)

            

            curoutput = cv2.cvtColor(curoutput, cv2.COLOR_RGB2BGR)
            out.write(np.uint8(curoutput))

            for i in range(1,config.prev_output_num):
                prev_output[config.prev_output_num-i] = prev_output[config.prev_output_num-i-1]
            prev_output[0] = curOutput

        cap.release()
        out.release()"""


def evaluate():
    print 'Evaluation Start'
    date = datetime.datetime.now().strftime('%Y_%m_%d/%H-%M')
    # directories
    mode_dir = config.root_dir + '{}'.format(tl.global_flag['mode'])
    ckpt_dir = mode_dir + '/fixed_ckpt'
    save_dir = mode_dir + '/result/'

    tl.files.exists_or_mkdir(ckpt_dir)
    tl.files.exists_or_mkdir(save_dir)

    sp_config.batch_size = 1
    
    # input
    test_video_list = np.array(sorted(tl.files.load_file_list(path = config.TEST.unstab_path, regx = '.*', printable = False)))

    for k in np.arange(len(test_video_list)):
        test_video_name = test_video_list[k]
        tf.reset_default_graph()
        cap = cv2.VideoCapture(config.TEST.unstab_path + test_video_name)
        fps = cap.get(5)
        resize_h = config.height
        resize_w = config.width
        # out_h = int(cap.get(4))
        # out_w = int(cap.get(3))
        out_h = resize_h
        out_w = resize_w
        print '[Outsize] h: ', out_h, ' w: ', out_w

        # refine_temp = np.ones((h, w))
        # refine_temp = refine_image(refine_temp)
        # [h, w] = refine_temp.shape[:2]
        total_frames = cap.get(7)
        fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
        base = os.path.basename(test_video_name)
        base_name = os.path.splitext(base)[0]

        out = cv2.VideoWriter(save_dir + base_name + '_out.avi', fourcc, fps, (2 * out_w, out_h))

        sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))
        with tf.variable_scope('input'):
            patches_prev_output = tf.placeholder('float32', [batch_size, config.prev_output_num, h, w, 3], name = 'input_frames_stab')
            patches_unstab = tf.placeholder('float32', [batch_size, h, w, 3], name = 'input_frames_unstab')
        # define model
        with tf.variable_scope('main_net') as scope:
            with tf.variable_scope('stab_net') as scope:
                with tf.variable_scope('2Dconv') as scope:
                    _,_,_,_,conv2dOut = UNet_down(patches_unstab, config.num_features, is_train=False, reuse = False, scope = scope)
                with tf.variable_scope('3Dconv') as scope:
                    conv3dOut = UNet_down_3D(patches_prev_output, is_train=False, reuse = False, scope = scope)
                with tf.variable_scope('merge') as scope:
                    convMerged = tf.concat([conv2dOut,conv3dOut],3)
                    print(convMerged)
                    convMerged = UNet_merged(convMerged, is_train=False, reuse=False, scope = scope)
                    homoTheta = dense_homo(convMerged,is_train=False, reuse=False, scope = scope);
                    #unetUpWOu4 = UNet_up_merged_without_u4(convMerged, is_train=False, reuse=False, scope = scope)

                    stl_projective = ProjectiveTransformer(sp_config.out_size)
                    HomoOutput = stl_projective.transform(patches_unstab, homoTheta )
                    model_output = HomoOutput # + unetUpWOu4

        # init session
        tl.layers.initialize_global_variables(sess)

        # load checkpoint
        tl.files.load_and_assign_npz_dict(name = ckpt_dir + '/{}.npz'.format(tl.global_flag['mode']), sess = sess)

        # read frame
        def refine_frame(frame):
            return cv2.resize(frame / 255., (resize_w, resize_h))

        def read_frame(cap):
            ref, frame = cap.read()
            if ref != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (out_w, out_h))

            return ref, frame

        frames_resize = []
        ref, frame = read_frame(cap)
        if ref == False:
            continue

        # initialize initial stablized frames with the first frame
        for i in np.arange(config.prev_output_num):
            frames_resize.append(refine_frame(frame))

        # read 2nd frame
        ref, frame = read_frame(cap)
        if ref == False:
            continue

        for frame_idx in range(1, int(total_frames)):
            prev_output = np.concatenate(frames_resize[0:len(frames_resize)], axis = 0)
            shape = prev_output.shape
            prev_output = prev_output.reshape((config.prev_output_num, shape[0] / config.seq_length, shape[1], shape[2]))

            # get stablilized frame
            feed_dict = {patches_unstab: np.expand_dims(refine_frame(frame), axis=0) ,patches_prev_output: np.expand_dims(prev_output,axis=0)}
            #{patches_source: train_frames, patch_source: np.expand_dims(frame, 0)}
            stab_result = sess.run(model_output, feed_dict)
            homoTheta_result = sess.run(homoTheta , feed_dict)
            print(homoTheta_result)
            stable_out = np.uint8(np.squeeze(stab_result * 255))

            # save
            output = np.concatenate((frame, stable_out), axis = 1)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(np.uint8(output))

            # replace stablized frame with unstable frame
            frames_resize[source_idx] = refine_frame(stable_out)

            # read an unstable frame
            del(frames_resize[0])
            ref, frame = read_frame(cap)
            if ref == False:
                break
            frames_resize.append(refine_frame(frame))


            # print log
            print('{}/{} {}/{}'.format(k, len(test_video_list), frame_idx, int(total_frames)))

        cap.release()
        out.release()


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

