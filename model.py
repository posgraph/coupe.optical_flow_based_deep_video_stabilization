import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from config import sp_config, config, log_config


def UNet_down(patch, num_features_out, is_train=False, reuse = False, scope = 'unet_down'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse = reuse) as vs:
        """ input layer """
        net_in = InputLayer(patch, name='input')
        """ conv1 """
        network = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        network = Conv2d(network, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        network = Conv2d(network, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2')
        d0 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2')
        d1 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_2')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_3')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_4')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_4')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_4')
        d2 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_2')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_3')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_4')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_4')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_4')
        d3 = network
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')                           
        """ conv5 """
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv5_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn5_1')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv5_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn5_2')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv5_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn5_3')
        network = PadLayer(network, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad5_4')
        network = Conv2d(network, n_filter=num_features_out, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv5_4')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn5_4')
        d4 = network

        return d0.outputs,d1.outputs,d2.outputs,d3.outputs,d4.outputs

def UNet_up(feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        d0 = InputLayer(feats[0], name='d0')
        d1 = InputLayer(feats[1], name='d1')
        d2 = InputLayer(feats[2], name='d2')
        d3 = InputLayer(feats[3], name='d3')
        d4 = InputLayer(feats[4], name='d4')

        n = UpSampling2dLayer(d4, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = ConcatLayer([n, d3], concat_dim = 3, name='u3/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = ConcatLayer([n, d2], concat_dim = 3, name='u2/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = ConcatLayer([n, d1], concat_dim = 3, name='u1/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = ConcatLayer([n, d0], concat_dim = 3, name='u0/concat')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')#
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c1')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b1')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad2')#
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c2')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b2')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad3')#pad1
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c3')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b3')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad4')#pad1
        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c4')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b4')#

        return n.outputs

def Localizer(feats, out_param_dim, is_train=False, reuse = False, is_tanh = False, scope = 'Localizer'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse = reuse) as vs:
        n = InputLayer(feats, name='input')

        n = Conv2d(n, 32, (3, 3), (2, 2), act=None, padding='VALID', W_init=w_init_relu, name='d1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d1/b1')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d2/b1')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d3/b1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d4/b1')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='d5/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='d5/b1')

        n = FlattenLayer(n, name='df/flatten1')
        n = DenseLayer(n, n_units = 256, act = tf.identity, name='df/dense1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b1')
        if is_tanh:
            n = DenseLayer(n, n_units = out_param_dim, act = tf.nn.tanh, name='df/dense2')
        else:
            n = DenseLayer(n, n_units = out_param_dim, act = tf.identity, name='df/dense2')

        return n.outputs


def UNet_down_3D_reshape(patch, is_train=False, reuse = False, scope = 'unet_down'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse = reuse) as vs:
        """ input layer """
        net_in = InputLayer(patch, name='input')
        """ conv1 """
        network = Conv3dLayer(net_in, shape=(3, 3, 3, 3, 32), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv1_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 32, 32), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv1_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool1')
        print(network)
        """ conv2 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 32, 64), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv2_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 64, 64), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv2_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool2')
        print(network)
        """ conv3 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 64, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_2')
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_3')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool3')
        print(network)
        """ conv4 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 256, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_2')
        network = Conv3dLayer(network, shape=(3, 3, 3, 256, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_3')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool4')
        print(network)

        return tf.reshape(network.outputs,[-1,22,22,512])


def UNet_down_3D(patch, is_train=False, reuse = False, scope = 'unet_down'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse = reuse) as vs:
        """ input layer """
        net_in = InputLayer(patch, name='input')
        """ conv1 """
        network = Conv3dLayer(net_in, shape=(3, 3, 3, 3, 32), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv1_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 32, 32), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv1_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool1')
        #print(network)
        """ conv2 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 32, 64), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv2_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 64, 64), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv2_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool2')
        #print(network)
        """ conv3 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 64, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_2')
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 128), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv3_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_3')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool3')
        #print(network)
        """ conv4 """
        network = Conv3dLayer(network, shape=(3, 3, 3, 128, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_1')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_1')
        network = Conv3dLayer(network, shape=(3, 3, 3, 256, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_2')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_2')
        network = Conv3dLayer(network, shape=(3, 3, 3, 256, 256), strides=(1,1,1,1,1), act=None, W_init=w_init_relu, padding='SAME', name='conv4_3')
        network = BatchNormLayer(network, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_3')
        network = MaxPool3d(network, filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME', name='pool4')
        #print(network)
        
        return tf.squeeze(network.outputs,[1])


def UNet_up_merged(feats, is_train=False, reuse=False, scope = 'unet_up'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:

        net_in = InputLayer(feats, name='input')

        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')#
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c1')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b1')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad2')#
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c2')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b2')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad3')#pad1
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c3')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b3')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad4')#pad1
        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c4')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b4')#

        return n.outputs


## for main_addhomo.py

def UNet_merged(feats, is_train=False, reuse=False, scope = 'unet_merged'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:

        net_in = InputLayer(feats, name='input')

        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad1')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad2')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u4/pad3')
        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u4/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u4/b3')

        return n.outputs

def dense_homo(feats,is_train=False, reuse=False, scope = 'dense_homo'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    customTanh = lambda x: tf.tanh(x/1000)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        net_in = InputLayer(feats, name='input')

        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2') 
        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_3')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        
        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1') 
        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2') 
        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_3')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        print(n.outputs)
        n = ReshapeLayer(n,[config.batch_size,-1],name='reshape')
        print(n.outputs)
        n = DenseLayer(n , n_units = 1000, act = lrelu, name='df/dense1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b1')
        n = DenseLayer(n, n_units = 1000, act = lrelu, name='df/dense2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b2')
        n = DenseLayer(n, n_units = 8, act = customTanh, name='df/dense3')

        #n = tf.multiply(n.outputs, tf.constant([[0.2,0.2,0.4,0.2,0.2,0.4,0.2,0.2]],dtype=tf.float32))
        #n = tf.multiply(n.outputs, tf.constant([[0.3,0.1,100,0.1,0.3,100,0.1,0.1]],dtype=tf.float32))
        #n = tf.multiply(n.outputs, tf.constant([[1,3,100,3,1,100,3,3]],dtype=tf.float32))
        
        n = tf.multiply(n.outputs, tf.constant([[0.3,0.1,170,0.1,0.3,170,0.01,0.01]],dtype=tf.float32))
        n = tf.add(n, tf.constant([[1,0,0,0,1,0,0,0]],dtype=tf.float32))
        print(n)
        n = tf.concat( [n,tf.constant([[1]],dtype=tf.float32)],axis=1 )
        print(n)
        n = tf.reshape(n,[-1,3,3])
        #n = tf.matrix_inverse(n)
        n = tf.reshape(n,[-1,9])
        n = n[:,0:8]

        #n = tf.multiply(n.outputs, tf.constant([[0,0,0,0,0,0,0,0]],dtype=tf.float32))
        #n = tf.add(n, tf.constant([[1,0,0.5,0,1,0.50,0,0]],dtype=tf.float32))
        
        #updateIdx = [ [b,x] for b in range(config.batch_size) for x in range(8) ]
        #updateIdxT = tf.constant(updateIdx)
        #updataVal = [ [-n[b,0]+n[b,0]/10+1] for b in range(config.batch_size), \
        #                [-n[1]+n[1]/10] for b in range(config.batch_size), \
        #                [-n[2]+n[2]/2] for b in range(config.batch_size),\
        #                [-n[3]+n[3]/10] for b in range(config.batch_size),\
        #                [-n[4]+n[4]/10+1] for b in range(config.batch_size),\
        #                [-n[5]+n[5]/2] for b in range(config.batch_size),\
        #                [-n[6]+n[6]/10] for b in range(config.batch_size),\
        #                [-n[7]+n[7]/10] for b in range(config.batch_size) ]
        #updataValT = tf.constant(updateVal)
        #updateShape = tf.constant([config.batch_size,8])
        #n= n + tf.scatter_nd(updateIdxT , updataValT, updateShape)
        ## n[0] = n[0]/10+1
        ## n[1] = n[1]/10
        ## n[2] = n[2]/2
        ## n[3] = n[3]/10
        ## n[4] = n[4]/10+1
        ## n[5] = n[5]/2
        ## n[6] = n[6]/10
        ## n[7] = n[7]/10
        ## n = tf.constant([n[:,0]/10+1,n[:,1]/10,n[:,2]/2,n[:,3]/10,n[:,4]/10+1,n[:,5]/2,n[:,6]/10,n[:,7]/10] , shape=[config.batch_size,8])
        #raise
        return n
def UNet_up_merged_without_u4(feats, is_train=False, reuse=False, scope = 'unet_up_without_u4'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(scope, reuse=reuse) as vs:

        net_in = InputLayer(feats, name='input')

        n = UpSampling2dLayer(net_in , (2, 2), is_scale = True, method = 1, align_corners=True, name='u3/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u3/pad3')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u3/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u3/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u2/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad1')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u2/pad3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u2/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u2/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u1/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u1/pad3')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u1/c3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u1/b3')

        n = UpSampling2dLayer(n, (2, 2), is_scale = True, method = 1, align_corners=True, name='u0/u')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='u0/pad_init')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='u0/c_init')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='u0/b_init')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad1')#
        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c1')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b1')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad2')#
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c2')#
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b2')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad3')#pad1
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c3')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b3')#
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='uf/pad4')#pad1
        n = Conv2d(n, 3, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='uf/c4')#c1
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='uf/b4')#

        return n.outputs


def discriminator(feats, is_train=False, reuse=False, scope = 'cnn'):
    g_init = None
    lrelu = lambda x: tf.nn.relu(x)
    identity = lambda x: x
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('cnn', reuse = False) as vs:
        n = InputLayer(feats, name='input')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='1_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='1_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1_2')
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME')
            
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='2_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='2_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2_2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='2_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2_3')
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_3')
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_3')
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_2')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_3')
        print(n.outputs)
        n = ReshapeLayer(n, [-1, 4*4*512*4])
        n = DenseLayer(n , n_units=2048, act=lrelu,name='1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1')
        n = DenseLayer(n , n_units=512, act=lrelu,name='2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2')
        n = DenseLayer(n , n_units=2, act=identity,name='3')
        n = tf.nn.sigmoid(n.outputs)

    return n


def flownetS(feats, batch_size, is_train=False, reuse=False, scope = 'flownetS' ):
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.1)
    identity = lambda x: x
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()

    def UpSampling2dLayer_(input, image_ref, method, align_corners, name):
        input = input.outputs
        size = tf.shape(image_ref)

        n = InputLayer(input, name = name + '_in')
        n = UpSampling2dLayer(n, size=[size[1], size[2]], is_scale = False, method = method, align_corners = align_corners, name = name)

        return n
    
    with tf.variable_scope(scope, reuse = reuse) as vs:
        n = InputLayer(feats, name='input')

        n = PadLayer(n, [[0, 0], [3, 3], [3, 3], [0, 0]], "constant")
        n = Conv2d(n, n_filter=64, filter_size=(7, 7), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='1')
        conv1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1')
        n = PadLayer(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=128, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='2')
        conv2 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2')
        n = PadLayer(conv2, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='3')
        conv3 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3')
        print(conv3.outputs)

        n = PadLayer(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_1')
        conv3_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_1')
        n = PadLayer(conv3_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='4')
        conv4 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4')
        n = PadLayer(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_1')
        conv4_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_1')
        n = PadLayer(conv4_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='5')
        conv5 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5')
        n = PadLayer(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_1')
        conv5_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_1')
        print(conv5_1.outputs)
        n = PadLayer(conv5_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='6')
        conv6 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6')
        print(conv6.outputs)
        n = PadLayer(conv6, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='6_1')
        conv6_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6_1')
        print(conv6_1.outputs)

        n = PadLayer(conv6_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow6 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict6')
        deconv5 = DeConv2dLayer(conv6_1,act=lrelu,shape=(4,4,512,1024),output_shape=(batch_size ,12,16,512),strides=(1,2,2,1),name='deconv5')
        #deconv5 = DeConv2dLayer(conv6_1,act=None,shape=(4,4,512,1024),output_shape=(batch_size ,12,16,512),strides=(1,2,2,1),name='deconv5')
        #deconv5 = BatchNormLayer(deconv5, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv5_bn')
        upsample_flow6to5 = DeConv2dLayer(predict_flow6,act=None,shape=(4,4,2,2),output_shape=(batch_size ,12,16,2),strides=(1,2,2,1),name='upsample6_5')
        concat5 = ConcatLayer([conv5_1,deconv5,upsample_flow6to5],concat_dim=3)

        n = PadLayer(concat5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow5 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict5')
        deconv4 = DeConv2dLayer(concat5,act=lrelu,shape=(4,4,256,1026),output_shape=(batch_size ,24,32,256),strides=(1,2,2,1),name='deconv4')
        #deconv4 = DeConv2dLayer(concat5,act=None,shape=(4,4,256,1026),output_shape=(batch_size ,24,32,256),strides=(1,2,2,1),name='deconv4')
        #deconv4 = BatchNormLayer(deconv4, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv4_bn')
        upsample_flow5to4 = DeConv2dLayer(predict_flow5,act= None,shape=(4,4,2,2),output_shape=(batch_size ,24,32,2),strides=(1,2,2,1),name='upsample5_4')
        concat4 = ConcatLayer([conv4_1,deconv4,upsample_flow5to4],concat_dim=3)

        n = PadLayer(concat4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow4 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict4')
        deconv3 = DeConv2dLayer(concat4,act=lrelu,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        #deconv3 = DeConv2dLayer(concat4,act=None,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        #deconv3 = BatchNormLayer(deconv3, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv3_bn')
        upsample_flow4to3 = DeConv2dLayer(predict_flow4,act= None,shape=(4,4,2,2),output_shape=(batch_size ,48,64,2),strides=(1,2,2,1),name='upsample4_3')
        concat3 = ConcatLayer([conv3_1,deconv3,upsample_flow4to3],concat_dim=3)

        n = PadLayer(concat3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow3 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict3')
        deconv2 = DeConv2dLayer(concat3,act=lrelu,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        #deconv2 = DeConv2dLayer(concat3,act=None,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        #deconv2 = BatchNormLayer(deconv2, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv2_bn')
        upsample_flow3to2 = DeConv2dLayer(predict_flow3,act= None,shape=(4,4,2,2),output_shape=(batch_size ,96,128,2),strides=(1,2,2,1),name='upsample3_2')
        concat2 = ConcatLayer([conv2,deconv2,upsample_flow3to2],concat_dim=3)

        n = PadLayer(concat2, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        #n = UpSampling2dLayer_(n, feats, method = 1, align_corners = True, name = 'upsample2_1')    # added
        predict_flow2 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict2')
        flow = predict_flow2.outputs * tf.constant(20.0)
        flow = tf.image.resize_bilinear(flow,tf.stack([384, 512]),align_corners=True)


    return {'predict_flow6': predict_flow6.outputs, 'predict_flow5': predict_flow5.outputs,'predict_flow4': predict_flow4.outputs,'predict_flow3': predict_flow3.outputs,'predict_flow2': predict_flow2.outputs,'flow': flow}



def flownetS_realdata(feats, batch_size, is_train=False, reuse=False, scope = 'flownetS' ):
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.1)
    identity = lambda x: x
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()

    def UpSampling2dLayer_(input, image_ref, method, align_corners, name):
        input = input.outputs
        size = tf.shape(image_ref)

        n = InputLayer(input, name = name + '_in')
        n = UpSampling2dLayer(n, size=[size[1], size[2]], is_scale = False, method = method, align_corners = align_corners, name = name)

        return n
    
    with tf.variable_scope(scope, reuse = reuse) as vs:
        n = InputLayer(feats, name='input')

        n = PadLayer(n, [[0, 0], [3, 3], [3, 3], [0, 0]], "constant")
        n = Conv2d(n, n_filter=64, filter_size=(7, 7), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='1')
        conv1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1')
        n = PadLayer(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=128, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='2')
        conv2 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2')
        n = PadLayer(conv2, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='3')
        conv3 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3')
        print(conv3.outputs)

        n = PadLayer(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_1')
        conv3_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_1')
        n = PadLayer(conv3_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='4')
        conv4 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4')
        n = PadLayer(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_1')
        conv4_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_1')
        n = PadLayer(conv4_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='5')
        conv5 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5')
        n = PadLayer(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_1')
        conv5_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_1')
        print(conv5_1.outputs)
        n = PadLayer(conv5_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='6')
        conv6 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6')
        print(conv6.outputs)
        n = PadLayer(conv6, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='6_1')
        conv6_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6_1')
        print(conv6_1.outputs)

        n = PadLayer(conv6_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow6 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict6')
        #deconv5 = DeConv2dLayer(conv6_1,act=lrelu,shape=(4,4,512,1024),output_shape=(batch_size ,12,16,512),strides=(1,2,2,1),name='deconv5')
        deconv5 = DeConv2dLayer(conv6_1,act=None,shape=(4,4,512,1024),output_shape=(batch_size ,12,16,512),strides=(1,2,2,1),name='deconv5')
        deconv5 = BatchNormLayer(deconv5, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv5_bn')
        upsample_flow6to5 = DeConv2dLayer(predict_flow6,act=None,shape=(4,4,2,2),output_shape=(batch_size ,12,16,2),strides=(1,2,2,1),name='upsample6_5')
        concat5 = ConcatLayer([conv5_1,deconv5,upsample_flow6to5],concat_dim=3)

        n = PadLayer(concat5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow5 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict5')
        #deconv4 = DeConv2dLayer(concat5,act=lrelu,shape=(4,4,256,1026),output_shape=(batch_size ,24,32,256),strides=(1,2,2,1),name='deconv4')
        deconv4 = DeConv2dLayer(concat5,act=None,shape=(4,4,256,1026),output_shape=(batch_size ,24,32,256),strides=(1,2,2,1),name='deconv4')
        deconv4 = BatchNormLayer(deconv4, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv4_bn')
        upsample_flow5to4 = DeConv2dLayer(predict_flow5,act= None,shape=(4,4,2,2),output_shape=(batch_size ,24,32,2),strides=(1,2,2,1),name='upsample5_4')
        concat4 = ConcatLayer([conv4_1,deconv4,upsample_flow5to4],concat_dim=3)

        n = PadLayer(concat4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow4 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict4')
        #deconv3 = DeConv2dLayer(concat4,act=lrelu,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        deconv3 = DeConv2dLayer(concat4,act=None,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        deconv3 = BatchNormLayer(deconv3, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv3_bn')
        upsample_flow4to3 = DeConv2dLayer(predict_flow4,act= None,shape=(4,4,2,2),output_shape=(batch_size ,48,64,2),strides=(1,2,2,1),name='upsample4_3')
        concat3 = ConcatLayer([conv3_1,deconv3,upsample_flow4to3],concat_dim=3)

        n = PadLayer(concat3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow3 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict3')
        #deconv2 = DeConv2dLayer(concat3,act=lrelu,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        deconv2 = DeConv2dLayer(concat3,act=None,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        deconv2 = BatchNormLayer(deconv2, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv2_bn')
        upsample_flow3to2 = DeConv2dLayer(predict_flow3,act= None,shape=(4,4,2,2),output_shape=(batch_size ,96,128,2),strides=(1,2,2,1),name='upsample3_2')
        concat2 = ConcatLayer([conv2,deconv2,upsample_flow3to2],concat_dim=3)

        n = PadLayer(concat2, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = UpSampling2dLayer_(n, feats, method = 1, align_corners = True, name = 'upsample2_1')    # added
        predict_flow2 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict2')
        flow = predict_flow2.outputs# * tf.constant(20.0)
        #flow = tf.image.resize_bilinear(flow,tf.stack([384, 512]),align_corners=True)


    return {'predict_flow6': predict_flow6.outputs, 'predict_flow5': predict_flow5.outputs,'predict_flow4': predict_flow4.outputs,'predict_flow3': predict_flow3.outputs,'predict_flow2': predict_flow2.outputs,'flow': flow}


def flownetS_pyramid(feats, batch_size, is_train=False, reuse=False, scope = 'flownetS' ):
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.1)
    identity = lambda x: x
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    featsSize = tf.shape(feats)
    featsSize = featsSize[0]

    def UpSampling2dLayer_(input, image_ref, method, align_corners, name):
        input = input.outputs
        size = tf.shape(image_ref)

        n = InputLayer(input, name = name + '_in')
        n = UpSampling2dLayer(n, size=[size[1], size[2]], is_scale = False, method = method, align_corners = align_corners, name = name)

        return n
    
    with tf.variable_scope(scope, reuse = reuse) as vs:
        n = InputLayer(feats, name='input')

        n = PadLayer(n, [[0, 0], [3, 3], [3, 3], [0, 0]], "constant")
        n = Conv2d(n, n_filter=64, filter_size=(7, 7), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='1')
        conv1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='1')
        n = PadLayer(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=128, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='2')
        conv2 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='2')
        n = PadLayer(conv2, [[0, 0], [2, 2], [2, 2], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(5, 5), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='3')
        conv3 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3')
        print(conv3.outputs)

        n = PadLayer(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='3_1')
        conv3_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='3_1')
        print(conv3_1.outputs)
        n = PadLayer(conv3_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='4')
        conv4 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4')
        print(conv4.outputs)
        n = PadLayer(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='4_1')
        conv4_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='4_1')
        print(conv4_1.outputs)
        n = PadLayer(conv4_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='5')
        conv5 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5')
        print(conv5.outputs)
        n = PadLayer(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='5_1')
        conv5_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='5_1')
        print(conv5_1.outputs)
        n = PadLayer(conv5_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(2, 2), act=None, W_init=w_init_relu, padding='VALID',name='6')
        conv6 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6')
        print(conv6.outputs)
        n = PadLayer(conv6, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = Conv2d(n, n_filter=1024, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID',name='6_1')
        conv6_1 = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init,name='6_1')
        print(conv6_1.outputs)

        n = PadLayer(conv6_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow6 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict6')
        #deconv5 = DeConv2dLayer(conv6_1,act=lrelu,shape=(4,4,512,1024),output_shape=(batch_size ,12,16,512),strides=(1,2,2,1),name='deconv5')
        deconv5 = DeConv2dLayer(conv6_1,act=None,shape=(4,4,512,1024),output_shape=(batch_size,12,16,512),strides=(1,2,2,1),name='deconv5')
        deconv5 = BatchNormLayer(deconv5, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv5_bn')
        upsample_flow6to5 = DeConv2dLayer(predict_flow6,act=None,shape=(4,4,2,2),output_shape=(batch_size,12,16,2),strides=(1,2,2,1),name='upsample6_5')
        concat5 = ConcatLayer([conv5_1,deconv5,upsample_flow6to5],concat_dim=3)

        n = PadLayer(concat5, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow5 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict5')
        predict_flow5 = tl.layers.ElementwiseLayer([predict_flow5, UpSampling2dLayer(predict_flow6, size = (12,16),is_scale=False), UpSampling2dLayer(predict_flow6, size = (12,16),is_scale=False)], combine_fn=tf.add)
        #deconv4 = DeConv2dLayer(concat5,act=lrelu,shape=(4,4,256,1026),output_shape=(batch_size ,24,32,256),strides=(1,2,2,1),name='deconv4')
        deconv4 = DeConv2dLayer(concat5,act=None,shape=(4,4,256,1026),output_shape=(batch_size,24,32,256),strides=(1,2,2,1),name='deconv4')
        deconv4 = BatchNormLayer(deconv4, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv4_bn')
        upsample_flow5to4 = DeConv2dLayer(predict_flow5,act= None,shape=(4,4,2,2),output_shape=(batch_size ,24,32,2),strides=(1,2,2,1),name='upsample5_4')
        concat4 = ConcatLayer([conv4_1,deconv4,upsample_flow5to4],concat_dim=3)

        n = PadLayer(concat4, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow4 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict4')
        predict_flow4 = tl.layers.ElementwiseLayer([predict_flow4, UpSampling2dLayer(predict_flow5, size = (24,32),is_scale=False), UpSampling2dLayer(predict_flow5, size = (24,32),is_scale=False)], combine_fn=tf.add)
        #deconv3 = DeConv2dLayer(concat4,act=lrelu,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        deconv3 = DeConv2dLayer(concat4,act=None,shape=(4,4, 128,770),output_shape=(batch_size,48,64,128),strides=(1,2,2,1),name='deconv3')
        deconv3 = BatchNormLayer(deconv3, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv3_bn')
        upsample_flow4to3 = DeConv2dLayer(predict_flow4,act= None,shape=(4,4,2,2),output_shape=(batch_size,48,64,2),strides=(1,2,2,1),name='upsample4_3')
        concat3 = ConcatLayer([conv3_1,deconv3,upsample_flow4to3],concat_dim=3)

        n = PadLayer(concat3, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        predict_flow3 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict3')
        predict_flow3 = tl.layers.ElementwiseLayer([predict_flow3,UpSampling2dLayer(predict_flow4, size = (48,64),is_scale=False), UpSampling2dLayer(predict_flow4, size = (48,64),is_scale=False)], combine_fn=tf.add)
        #deconv2 = DeConv2dLayer(concat3,act=lrelu,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        deconv2 = DeConv2dLayer(concat3,act=None,shape=(4,4, 64,386),output_shape=(batch_size,96,128,64),strides=(1,2,2,1),name='deconv2')
        deconv2 = BatchNormLayer(deconv2, act=lrelu, is_train = is_train, gamma_init = g_init,name='deconv2_bn')
        upsample_flow3to2 = DeConv2dLayer(predict_flow3,act= None,shape=(4,4,2,2),output_shape=(batch_size,96,128,2),strides=(1,2,2,1),name='upsample3_2')
        concat2 = ConcatLayer([conv2,deconv2,upsample_flow3to2],concat_dim=3)

        n = PadLayer(concat2, [[0, 0], [1, 1], [1, 1], [0, 0]], "constant")
        n = UpSampling2dLayer_(n, feats, method = 1, align_corners = True, name = 'upsample2_1')    # added
        #n = tl.layers.ElementwiseLayer([n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n], combine_fn=tf.add) 
        predict_flow2 = Conv2d(n, n_filter=2, filter_size=(3, 3), act=None, strides=(1, 1), W_init=w_init_relu, padding='VALID',name='predict2')
        upprevlayer =  UpSampling2dLayer(predict_flow3, size = (382,510),is_scale=False);
        predict_flow2 = tl.layers.ElementwiseLayer([predict_flow2,upprevlayer,upprevlayer,upprevlayer,upprevlayer,upprevlayer,upprevlayer,upprevlayer,upprevlayer], combine_fn=tf.add) 
        #predict_flow2 = tl.layers.ElementwiseLayer([predict_flow2,upprevlayer,upprevlayer,upprevlayer,upprevlayer], combine_fn=tf.add) 
        flow = predict_flow2.outputs# * tf.constant(20.0)
        #flow = tf.image.resize_bilinear(flow,tf.stack([384, 512]),align_corners=True)


    return {'predict_flow6': predict_flow6.outputs, 'predict_flow5': predict_flow5.outputs,'predict_flow4': predict_flow4.outputs,'predict_flow3': predict_flow3.outputs,'predict_flow2': predict_flow2.outputs,'flow': flow}



def flownetS_prehomo(feats,batch_size,is_train=False, reuse=False, scope = 'dense_homo'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    customTanh = lambda x: tf.tanh(x/1000)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        net_in = InputLayer(feats, name='input')

        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_3')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_3')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_1')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_2')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_3')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_1')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_2')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_3')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

        n = ReshapeLayer(n,[batch_size,-1],name='reshape')
        
        n = DenseLayer(n , n_units = 256, act = lrelu, name='df/dense1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b1')
        n = DenseLayer(n, n_units = 128, act = lrelu, name='df/dense2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b2')
        n = DenseLayer(n, n_units = 8, act = None, name='df/dense3')

    return n.outputs


def flownetS_variableWeight(feats,batch_size,is_train=False, reuse=False, scope = 'dense_homo'):
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()
    w_init_sigmoid = tf.contrib.layers.xavier_initializer()
    #g_init = tf.random_normal_initializer(1., 0.02)
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    customTanh = lambda x: tf.tanh(x/1000)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        net_in = InputLayer(feats, name='input')

        n = PadLayer(net_in, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_1')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_1')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_2')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad1_3')
        n = Conv2d(n, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv1_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn1_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_1')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_2')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad2_3')
        n = Conv2d(n, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv2_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn2_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_1')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_2')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad3_3')
        n = Conv2d(n, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv3_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn3_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_1')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_1') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_2')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_2') 
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "Symmetric", name='pad4_3')
        n = Conv2d(n, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=None, W_init=w_init_relu, padding='VALID', name='conv4_3')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='bn4_3') 
        n = MaxPool2d(n, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')

        n = ReshapeLayer(n,[batch_size,-1],name='reshape')
        
        n = DenseLayer(n , n_units = 256, act = lrelu, name='df/dense1')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b1')
        n = DenseLayer(n, n_units = 128, act = lrelu, name='df/dense2')
        n = BatchNormLayer(n, act=lrelu, is_train = is_train, gamma_init = g_init, name='df/b2')
        n = DenseLayer(n, n_units = 8, act = None, name='df/dense3')

    return n.outputs


def flownetS_288(input, is_train = False, reuse = False, scope = 'flownetS'):
    g_init = None
    lrelu = lambda x: tl.act.lrelu(x, 0.1)
    w_init_relu = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope(scope, reuse = reuse):
        batch_size = tf.shape(input)[0]
        n = InputLayer(input, name = 'input')

        # DECODER
        n = PadLayer(n, [[0, 0], [3, 3], [3, 3], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 32, filter_size = (7, 7), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '1')
        down0 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '1')
        n = PadLayer(down0, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 32, filter_size = (5, 5), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '2')
        down1 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '2')
        n = PadLayer(down1, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 64, filter_size = (5, 5), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '3')
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '3')

        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 64, filter_size = (3, 3), strides = (1, 1), act = None, W_init = w_init_relu, padding = 'VALID', name = '3_1')
        down2 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '3_1')

        n = PadLayer(down2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 128, filter_size = (3, 3), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '4')
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '4')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 128, filter_size = (3, 3), strides = (1, 1), act = None, W_init = w_init_relu, padding = 'VALID', name = '4_1')
        down3 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '4_1')

        n = PadLayer(down3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 256, filter_size = (3, 3), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '5')
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '5')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 256, filter_size = (3, 3), strides = (1, 1), act = None, W_init = w_init_relu, padding = 'VALID', name = '5_1')
        down4 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '5_1')

        n = PadLayer(down4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 512, filter_size = (3, 3), strides = (2, 2), act = None, W_init = w_init_relu, padding = 'VALID', name = '6')
        n = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '6')
        n = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        n = Conv2d(n, n_filter = 1024, filter_size = (3, 3), strides = (1, 1), act = None, W_init = w_init_relu, padding = 'VALID', name = '6_1')
        down5 = BatchNormLayer(n, act = lrelu, is_train = is_train, gamma_init = g_init, name = '6_1')

        # ENCODER
        n = PadLayer(down5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow6 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict6')
        deconv5 = UpSampling2dLayer(down5, size = (9, 16), is_scale = False, method = 0, align_corners = True, name = 'upsample_5')
        deconv5 = PadLayer(deconv5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_5/pad1')
        deconv5 = Conv2d(deconv5, 512, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_5/c1')
        deconv5 = BatchNormLayer(deconv5, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_5/b1')
        upsample_flow6to5 = UpSampling2dLayer(predict_flow6, size = (9, 16), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow6to5')
        upsample_flow6to5 = PadLayer(upsample_flow6to5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow6to5/pad1')
        upsample_flow6to5 = Conv2d(upsample_flow6to5, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow6to5/c1')

        concat5 = ConcatLayer([down4,deconv5,ElementwiseLayer([upsample_flow6to5, upsample_flow6to5], tf.add)],concat_dim=3)

        n = PadLayer(concat5, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow5 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict5')
        deconv4 = UpSampling2dLayer(concat5, size = (18, 32), is_scale = False, method = 0, align_corners = True, name = 'upsample_4')
        deconv4 = PadLayer(deconv4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_4/pad1')
        deconv4 = Conv2d(deconv4, 256, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_4/c1')
        deconv4 = BatchNormLayer(deconv4, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_4/b1')
        upsample_flow5to4 = UpSampling2dLayer(predict_flow5, size = (18, 32), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow5to4')
        upsample_flow5to4 = PadLayer(upsample_flow5to4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow5to4/pad1')
        upsample_flow5to4 = Conv2d(upsample_flow5to4, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow5to4/c1')

        concat4 = ConcatLayer([down3,deconv4, ElementwiseLayer([upsample_flow5to4, upsample_flow5to4], tf.add)],concat_dim=3)

        n = PadLayer(concat4, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow4 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict4')
        deconv3 = UpSampling2dLayer(concat4, size = (36, 64), is_scale = False, method = 0, align_corners = True, name = 'upsample_3')
        deconv3 = PadLayer(deconv3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_3/pad1')
        deconv3 = Conv2d(deconv3, 128, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_3/c1')
        deconv3 = BatchNormLayer(deconv3, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_3/b1')
        upsample_flow4to3 = UpSampling2dLayer(predict_flow4, size = (36, 64), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow4to3')
        upsample_flow4to3 = PadLayer(upsample_flow4to3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow4to3/pad1')
        upsample_flow4to3 = Conv2d(upsample_flow4to3, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow4to3/c1')

        concat3 = ConcatLayer([down2,deconv3,ElementwiseLayer([upsample_flow4to3, upsample_flow4to3], tf.add)],concat_dim=3)

        n = PadLayer(concat3, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow3 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict3')
        deconv2 = UpSampling2dLayer(concat3, size = (72, 128), is_scale = False, method = 0, align_corners = True, name = 'upsample_2')
        deconv2 = PadLayer(deconv2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_2/pad1')
        deconv2 = Conv2d(deconv2, 64, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_2/c1')
        deconv2 = BatchNormLayer(deconv2, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_2/b1')
        upsample_flow3to2 = UpSampling2dLayer(predict_flow3, size = (72, 128), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow3to2')
        upsample_flow3to2 = PadLayer(upsample_flow3to2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow3to2/pad1')
        upsample_flow3to2 = Conv2d(upsample_flow3to2, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow3to2/c1')

        concat2 = ConcatLayer([down1,deconv2,ElementwiseLayer([upsample_flow3to2, upsample_flow3to2], tf.add)],concat_dim=3)

        n = PadLayer(concat2, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow2 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict2')
        deconv1 = UpSampling2dLayer(concat2, size = (144, 256), is_scale = False, method = 0, align_corners = True, name = 'upsample_1')
        deconv1 = PadLayer(deconv1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_1/pad1')
        deconv1 = Conv2d(deconv1, 32, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_1/c1')
        deconv1 = BatchNormLayer(deconv1, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_1/b1')
        upsample_flow2to1 = UpSampling2dLayer(predict_flow2, size = (144, 256), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow2to1')
        upsample_flow2to1 = PadLayer(upsample_flow2to1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow2to1/pad1')
        upsample_flow2to1 = Conv2d(upsample_flow2to1, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow2to1/c1')

        concat1 = ConcatLayer([down0,deconv1, ElementwiseLayer([upsample_flow2to1, upsample_flow2to1], tf.add)],concat_dim=3)

        n = PadLayer(concat1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
        predict_flow1 = Conv2d(n, n_filter = 2, filter_size = (3, 3), act = None, strides = (1, 1), W_init = w_init_relu, padding = 'VALID', name = 'predict1')
        deconv0 = UpSampling2dLayer(concat1, size = (288, 512), is_scale = False, method = 0, align_corners = True, name = 'upsample_0')
        deconv0 = PadLayer(deconv0, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_0/pad1')
        deconv0 = Conv2d(deconv0, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_0/c1')
        deconv0 = BatchNormLayer(deconv0, act=lrelu, is_train = is_train, gamma_init = g_init, name='upsample_0/b1')
        upsample_flow1to0 = UpSampling2dLayer(predict_flow1, size = (288, 512), is_scale = False, method = 0, align_corners = True, name = 'upsample_flow1to0')
        upsample_flow1to0 = PadLayer(upsample_flow1to0, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='upsample_flow1to0/pad1')
        upsample_flow1to0 = Conv2d(upsample_flow1to0, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='upsample_flow1to0/c1')

        concat0 = ConcatLayer([deconv0, ElementwiseLayer([upsample_flow1to0, upsample_flow1to0], tf.add)],concat_dim=3)

        flow = UpSampling2dLayer(concat0, size = (288, 512), is_scale = False, method = 0, align_corners = True, name = 'flow')
        flow = PadLayer(flow, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='flow/pad1')
        flow = Conv2d(flow, 16, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='flow/c1')
        flow = BatchNormLayer(flow, act=lrelu, is_train = is_train, gamma_init = g_init, name='flow/b1')
        flow = PadLayer(flow, [[0, 0], [1, 1], [1, 1], [0, 0]], 'Symmetric', name='flow/pad2')
        flow = Conv2d(flow, 2, (3, 3), (1, 1), act=None, padding='VALID', W_init=w_init_relu, name='flow/c2')

        flow = flow.outputs

    return {'flow6': predict_flow6.outputs, 'flow5': predict_flow5.outputs,\
    'flow4': predict_flow4.outputs, 'flow3': predict_flow3.outputs,\
    'flow2': predict_flow2.outputs, 'flow1': predict_flow1.outputs,\
    'flow': flow}
