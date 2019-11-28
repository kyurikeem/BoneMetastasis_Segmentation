"""
#---------------------------*[Description]*---------------------------------#
#            * Make pickle file for low HU value image data
#---------------------------------------------------------------------------#
"""
import os
import re
import tensorflow as tf
from scipy import ndimage as nd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pickle

import Make_cv_pickle_List
import lytic_HU_3cto3v_Loader_patch

flags = tf.app.flags
FLAGS = flags.FLAGS

data_Par_dir = '/data/Data_Set/BM_CV'
training_batch_size = 1
flags.DEFINE_string("data_dir", data_Par_dir, 'path of dataset')
flags.DEFINE_integer("training_batch_size", training_batch_size, "batch size for training data set")

def pause():
    input('Press the <Enter> key to continue...')

def int_text_distinguisher(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''

    return [int_text_distinguisher(c) for c in re.split('(\d+)', text)]

Sub_Name_list = []
for Par_dir, Sub_Name, File_name in os.walk(FLAGS.data_dir):
    if Sub_Name:
        for itr in range(len(Sub_Name)):
            if 'v0.2' in Sub_Name[itr].split('_'):
                Sub_Name_list.append(Sub_Name[itr])
train_File_path_list = []
for i in range(len(Sub_Name_list)):
    train_path = os.path.join(FLAGS.data_dir, Sub_Name_list[i])
    train_list = os.listdir(train_path)
    train_File_path_list.append(train_list)
train_File_path_list = sum(train_File_path_list, [])

main_patch_pickle_list = []
for itr2 in range(len(train_File_path_list)):
    if [words for words in ['crop(0, 50)'] if words in train_File_path_list[itr2].split('_')]:
        main_patch_pickle_list.append(train_File_path_list[itr2])
    else:
        pass

main_patch_pickle_list.sort(key=natural_keys)
main_patch_pickle_list = [main_patch_pickle_list[4], main_patch_pickle_list[1], main_patch_pickle_list[2], main_patch_pickle_list[3]]

train_records = Make_cv_pickle_List.load_pickle(train_path, main_patch_pickle_list)
print('len(train_records):', len(train_records))

resize_option = {'resize': False }

low_HU_records = []

train_batch_loader = lytic_HU_3cto3v_Loader_patch.data_loader(train_records, resize_option)
MAX_ITERATION = len(train_records)

for itr in range(MAX_ITERATION):
    train_batch_pre_images, train_batch_main_images, train_batch_post_images, train_batch_main_annotation, file_list = train_batch_loader.get_next_batch(FLAGS.training_batch_size)
    # print(file_list)

    batch_main_training_image = np.squeeze(train_batch_main_images)
    batch_main_training_annotation = np.squeeze(train_batch_main_annotation)
    batch_main_multiple_image = batch_main_training_image * batch_main_training_annotation

    lbl, nlbl = nd.label(batch_main_multiple_image)
    lbls = np.arange(1, nlbl + 1)
    get_median_value = nd.labeled_comprehension(batch_main_multiple_image, lbl, lbls, np.median, float, -1)
    print('Median_HU_Value:',get_median_value)

    if np.all(get_median_value < 1300):
        print('\r')
        print('==============================')
        print(file_list)
        print(get_median_value)
        print('==============================')
        print('\r')
        file_list = sum(file_list, [])
        low_HU_records.append(file_list)

with open('Osteolyitc_low_value_HU_fold1.pickle', 'wb') as f:
    pickle.dump(low_HU_records, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('Osteolyitc_low_value_HU_fold1.pickle', 'rb') as f:
    result = pickle.load(f)

print('Pickle file Ready!')
print(result)
print(len(result))







