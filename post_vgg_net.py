"""
#------------------------*[Record of version]*---------------------------#
#                             VGG-19 Post
#------------------------------------------------------------------------#
"""

import TensorFlowUtils_offline as utils
import numpy as np
import tensorflow as tf


def pause():
    input("Press the <Enter> key to continue...")


def vgg_net(image, layer_set, debuging_option):
    name_of_layer = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',  

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    # imagenet-vgg-verydeep-19.mat의 layers 필드안에 37번째 까지가 pool5. Conv, relu, pooling 수행하는 layer는 pool5 까지고,
    # 그 뒤에 fc6, fc7, fc8 필드가 있어서 19개 layer가 존재.

    net = {}
    current_data = image
    current_data_shape = image.get_shape()
    # no_of_images = current_data_shape[0].value
    # image_row = current_data_shape[1].value
    # image_column = current_data_shape[2].value
    current_data_channel = current_data_shape[3].value


    for i, name in enumerate(name_of_layer):
        layer_kind = name[:4] #layer_kind will be conv, relu, pool.

        if layer_kind == 'conv':
            kernels, bias = layer_set[i]['weights'][0][0][0]

            # since the dimension of matconvnet weights are [width, height, in_channels, out_channels],
            # need to change dimension [height, width, in_channels, out_channels] for TensorFlow.
            original_kernel = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name='post_model_' + name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name='post_model_' + name + "_b") #reshape(-1)은 flatten 역할

            # np.transpose() help to move 1th dimension value to 0th place and 0th value to 1th place.
            # 3vgg에 3채널로 들어갔던 모습을 수정하여 입력 1채널에 따른 kernel을 1채널을 만들도록함.
            if current_data_channel == 1:
                # kernels_for_1channel = original_kernel[:][:, :][:, :, :1][:, :, :, :]

                kernels_for_b_channel = original_kernel[0][:, :, np.newaxis, :]
                kernels_for_r_channel = original_kernel[1][:, :, np.newaxis, :]
                kernels_for_g_channel = original_kernel[2][:, :, np.newaxis, :]
                kernels_for_1channel = kernels_for_r_channel + kernels_for_g_channel + kernels_for_b_channel

                current_data = utils.conv2d_basic(current_data, kernels_for_1channel, bias)

            else:
                kernels_for_3channel = original_kernel
                current_data = utils.conv2d_basic(current_data, kernels_for_3channel, bias)

            current_data_shape = current_data.get_shape()
            current_data_channel = current_data_shape[3].value

        elif layer_kind == 'relu':
            current_data = tf.nn.relu(current_data, name='post_model_' + name)

            if debuging_option:
                utils.add_activation_summary(current_data)

        elif layer_kind == 'pool':
            current_data = utils.avg_pool_2x2(current_data, name='post_model_' + name)

        net[name] = current_data #dict 형태로 net에 저장.

    return net







