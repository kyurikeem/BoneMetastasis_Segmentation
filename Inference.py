"""
#------------------------------------------------------------------------#
#------------------------*[Record of version]*---------------------------#
#------------------------------------------------------------------------#
#     Date         |   Name     |        Version description             #
#------------------------------------------------------------------------#
# Fri.Mar.09.2018       GB         Up-Sampling, Deconvolution layer
#------------------------------------------------------------------------#
"""

import TensorFlowUtils_offline as utils
import tensorflow as tf
import pre_vgg_net_GB
import main_vgg_net_GB
import post_vgg_net_GB


def pause():
    input('Press the <Enter> key to continue...')

def inference_process(pre_image, main_image, post_image, weights, normal_values, keep_prob, num_of_classes, debug_mode):
    """
    :param image:
    :param weights:
    :param keep_prob:
    :param num_of_classes:
    :param debug_mode:
    :param normal_values:
    :return:
    """

    with tf.variable_scope('inference'):
        pre_image_net = pre_vgg_net_GB.vgg_net(pre_image, weights, debug_mode)
        main_image_net = main_vgg_net_GB.vgg_net(main_image, weights, debug_mode)
        post_image_net = post_vgg_net_GB.vgg_net(post_image, weights, debug_mode)

        pre_conv_layer_out = pre_image_net['conv5_3']
        main_conv_layer_out = main_image_net['conv5_3']
        post_conv_layer_out = post_image_net['conv5_3']

        layer_out_fuse1 = tf.add(pre_conv_layer_out, main_conv_layer_out, name='layer_out_fuse1')
        layer_out_fuse2 = tf.add(layer_out_fuse1, post_conv_layer_out)

        pool5 = utils.max_pool_2x2(layer_out_fuse2)

        W6 = utils.weight_variable([7, 7, 512, 4096], name='W6')
        b6 = utils.bias_variable([4096], name='b6')
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name='relu6')

        if debug_mode:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name='W7')
        b7 = utils.bias_variable([4096], name='b7')
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name='relu7')
        if debug_mode:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)


        W8 = utils.weight_variable([1, 1, 4096, num_of_classes], name='W8')
        b8 = utils.bias_variable([num_of_classes], name='b8')
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        annotation_prediction1 = tf.argmax(conv8, axis=3, name='prediction1')
        #conv8 output의 채널 (num_of_classes) 중 가장 높은 값을 select!

        # 여기서부터 deconvolution 수행을 위해 setting
        # Output upscaling to the actual image size
        deconv_shape1 = main_image_net['pool4'].get_shape()# pool4 output shape을 가져옴 #12th-layer임

        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, num_of_classes], name='W_t1')

        b_t1 = utils.bias_variable([deconv_shape1[3].value], name='b_t1')
        deconv_image_1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(main_image_net["pool4"]))

        fuse_1 = tf.add(deconv_image_1, main_image_net['pool4'], name='fuse_1')

        deconv_shape2 = main_image_net['pool3'].get_shape()


        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name='W_t2')
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name='b_t2')
        deconv_image_2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(main_image_net['pool3']))
        fuse_2 = tf.add(deconv_image_2, main_image_net['pool3'], name='fuse_2')


        shape = tf.shape(main_image)  # What is tf.shape function
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], num_of_classes])
        W_t3 = utils.weight_variable([16, 16, num_of_classes, deconv_shape2[3].value], name='W_t3')
        b_t3 = utils.bias_variable([num_of_classes], name='b_t3')
        deconv_image_3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8) #stride = 8 삭제함 ->2로


        annotation_pred = tf.argmax(deconv_image_3, axis=3, name='prediction')  # What is dimension here?

    return tf.expand_dims(annotation_pred, dim=3), deconv_image_3


