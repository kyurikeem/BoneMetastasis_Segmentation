"""
#========================================================================#
#------------------------*[Record of version]*---------------------------#
#------------------------------------------------------------------------#
#     Date         |                Version description                  #
#------------------------------------------------------------------------#
#                               spine centered image crop
#------------------------------------------------------------------------#
"""
#  Crop Augmentation
#  Patch size 200x200

import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt


def pause():
    input('Press the <Enter> key to continue...')


def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])




def random_crop_center(pre, main, post, annot ,cropx, cropy, crop_option):

    center_point_option = crop_option
    # center_point_option = {'point_value': [(0,0), (0,50), (0,100), (-50,50), (50,50),
    # (50,0), (50,100), (-50,0), (-50,100), (25,25), (25,75), (-25,25), (-25,75)]}

    """ cropy,cropx: 200 200
    
        Spine Center Point(250, 350) = (0, 50)    / startx, starty: 156 206 
        Center Point Move Up = (0, 0)             / startx, starty: 156 156
        Center Point Move Down = (0, 100)         / startx, starty: 156 256
        Center Point Move Left = (-50, 50)        / startx, starty: 106 206
        Center Point Move Right = (50, 50)        / startx, starty: 206 206  
                                  (50, 0)         / startx, starty: 206 156 
                                  (50, 100)       / startx, starty: 206 256 
                                  (-50, 0)        / startx, starty: 106 156 
                                  (-50, 100)      / startx, starty: 106 256 
                                  (25, 25)        / startx, starty: 181 181
                                  (25, 75)        / startx, starty: 181 231
                                  (-25, 25)       / startx, starty: 131 181   
                                  (-25, 75)       / startx, starty: 131 231
        """

    # print('center_point_option:', random.choice(center_point_option['point_value']))
    x,y,z = main.shape


    # center_point_option = random.choice(center_point_option['point_value'])
    startx = (x // 2 - (cropx // 2)) + center_point_option[0]
    starty = (y // 2 - (cropy // 2)) + center_point_option[1]

    # print('startx, starty:',startx ,starty)
    # print('cropy,cropx:', cropy,cropx)
    # pause()

    PR = pre[starty:starty+cropy,startx:startx+cropx]
    M = main[starty:starty+cropy,startx:startx+cropx]
    PO = post[starty:starty+cropy,startx:startx+cropx]
    A = annot[starty:starty+cropy,startx:startx+cropx]

    # print(np.unique(M))
    # print(np.unique(A))
    # annotation_mask= (M*A)
    # print('========== main x annot ==========')
    # print(np.unique(annotation_mask))
    # # print(np.count_nonzero(annotation_mask))
    # annotation_mask_nonzero = annotation_mask[annotation_mask != 0]
    # print('========== main x annot_nonzero ==========')
    # print(np.unique(annotation_mask_nonzero))
    # print(np.mean(annotation_mask_nonzero))
    # print(M.shape, A.shape)
    # # if np.count_nonzero(A) == 0:
    # print('Lets plot!')
    # flip_fig = plt.figure('before flip')
    # sub_before = flip_fig.add_subplot(1, 5, 1)
    # sub_before.imshow(PR[:][:, :][:, :, 0], cmap=plt.cm.bone)
    # sub_before.axis('off')
    # sub_after = flip_fig.add_subplot(1, 5, 2)
    # sub_after.imshow(M[:][:, :][:, :, 0], cmap=plt.cm.bone)
    # sub_after.axis('off')
    # sub_after = flip_fig.add_subplot(1, 5, 3)
    # sub_after.imshow(PO[:][:, :][:, :, 0], cmap=plt.cm.bone)
    # sub_after.axis('off')
    # sub_after = flip_fig.add_subplot(1, 5, 4)
    # sub_after.imshow(A[:][:, :][:, :, 0], cmap=plt.cm.bone)
    # sub_after.axis('off')
    # sub_after = flip_fig.add_subplot(1, 5, 5)
    # sub_after.imshow(annotation_mask[:][:, :][:, :, 0], cmap=plt.cm.bone)
    # sub_after.axis('off')
    # flip_fig.show()
    # pause()


    return pre[starty:starty+cropy,startx:startx+cropx], main[starty:starty+cropy,startx:startx+cropx],post[starty:starty+cropy,startx:startx+cropx],annot[starty:starty+cropy,startx:startx+cropx]

