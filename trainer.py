"""
#------------------------*[Record of version]*---------------------------#
#            GradientDescentOptimizer, AdamOptimizer Optional
#------------------------------------------------------------------------#
"""

import tensorflow as tf
import TensorFlowUtils_offline as utils

def train(loss_val, var_list, learning_rate, debug_mode):
    # optimizer = tf.train.GradientDescentOptimizer (learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    if debug_mode:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)

    return optimizer.apply_gradients(grads)