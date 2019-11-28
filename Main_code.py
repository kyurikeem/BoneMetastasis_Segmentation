"""
#===========================================================================#
#---------------------------*[Description]*---------------------------------#
#           * This is the main code for BM 3ch to 3 vgg model
#===========================================================================#
#------------------------*[Record of version]*------------------------------#
#---------------------------------------------------------------------------#
#     Date         |   Name     |           Version description             #
#---------------------------------------------------------------------------#
# 2018.01.21                              5 fold Cross Validation
#---------------------------------------------------------------------------#
"""
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

#--------Python in-built modules----------#
import tensorflow as tf
import numpy as np
import datetime
import random
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
from PIL import Image
import os
import re
import pickle
#-----------------------------------------#


#------------import modules----------------#
import ReadMakeList_3cto3v_BoneMeta_dcm_nii as BoneMeta_3to1_data_list
import ReadMakeList_per_patient as BoneMeta_patient_list
import TensorFlowUtils_offline as utils
import Inference
import Weights_Loader
import trainer as opti_trainer
import Make_cv_pickle_List
import Bone_Meta_Data_3cto3v_Loader_512_size
import Bone_Meta_Data_3cto3v_Loader_patch
#------------------------------------------#

'''
=======================
0. GPU Set
=======================
'''
flags = tf.app.flags
FLAGS = flags.FLAGS
gpu_number = str(5)
start_fold = 5
flags.DEFINE_string('GPU_number', gpu_number, 'Select the specific GPU device.')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.GPU_number   #'CUDA_VISIBLE_DEVICES' 프로세스에 표시되는 모든 GPU의 모든 GPU 메모리 매핑

# session에서 필요로 하는 메모리를 최소한으로 할당한 후 필요할 때마다 점진적으로 늘려가면서 메모리를 추가할당
# sess=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True)))

'''
=======================
1. Parameters
=======================
'''
log_txt_front_name='BM_3to1_final_data_cv'
#-------------------------------------------------------------#
#### Lately Update Bone meta Data File #####
data_Par_dir = '/data/Data_Set/BM_CV'
data_addition_dir = '/data/Data_Set/BM_CV/Osteolytic_pickle'
model_dir = "/data/Model_Set/VGG_19_uptodated_model"
model_file_name = 'imagenet-vgg-verydeep-19.mat'
name_of_optimizer = 'AdamOptimizer'
mode = 'train' #'Mode train/visualize'
training_batch_size = 8
validation_batch_size = 1
learning_rate = 0.00001
num_of_epochs = 10 #epoch 수 만큼 돌아감.
num_of_classes = 2
drop_out_value = 0.5
loss_name = 'sparse_softmax_cross_entropy_with_logits'
# loss_name = 'weighted_loss_for_cost_sensitive_learning'

'''
=======================
2-1. Augmentation Option
=======================
'''
# print('2-1. Augmentation Step')
# print('2-1. Augmentation Step\r',file=log_text)

augmentation_option = {'augmentation': True}
###augmentation option###
resize_size = 512
# resize_size = 256
value_to_add_list = np.arange(-100, 101, step=50)
upside_crop_value_list = np.arange(1, 51, step=10)
downside_crop_value_list = np.arange(463, 513, step=10)
##################################
resize_option = {'resize': False, 'resize_size': resize_size}
LR_reverse_option = {'LR_reverse': True, 'LR_case': [0, 1]}  # x2
add_to_pixel_option = {'add_to_pixel': True, 'add_value': value_to_add_list}
crop_option = {'crop_option': True, 'crop_case': [0, 1]}
upside_crop = {'upside_crop_value': upside_crop_value_list}
downside_crop = {'downside_crop_value': downside_crop_value_list}


#### Do not need to change the lines below. ####
logs_dir_path = data_Par_dir + '/' + log_txt_front_name + '_' + str(datetime.date.today()) + '_logs_GPU_' + gpu_number
log_txt_name = log_txt_front_name + '_GPU_' + gpu_number + '_' + str(datetime.date.today()) + '.txt'
flags.DEFINE_float('start_fold', start_fold, 'The ratio of drop out')
flags.DEFINE_integer('epochs', num_of_epochs, 'The number of epochs')
flags.DEFINE_integer('classes',num_of_classes, 'The number of classes')
flags.DEFINE_float('drop_out', drop_out_value, 'The ratio of drop out')
flags.DEFINE_string("logs_dir", logs_dir_path, "path of a logs directory")
flags.DEFINE_string("data_dir", data_Par_dir, 'path of dataset')
flags.DEFINE_string("data_addition_dir", data_addition_dir, 'path of Added dataset')
flags.DEFINE_string("model_dir", model_dir, "Path to vgg model mat file")
flags.DEFINE_string('model_file_name', model_file_name, 'File name of VGG-19 model')
flags.DEFINE_string('mode', mode, 'Mode train/visualize')
flags.DEFINE_integer("training_batch_size", training_batch_size, "batch size for training data set")
flags.DEFINE_integer('validation_batch_size', validation_batch_size, 'batch size for validation data set')
flags.DEFINE_float("learning_rate", learning_rate, 'Learning rate')
flags.DEFINE_bool("debug_mode", False, "Debug mode: True/False")

'''
========================
!!!!!!!!! MAIN !!!!!!!!!
========================
'''
def main(argv_None):
    ######log dir making######
    try:
        if not (os.path.isdir(FLAGS.logs_dir)):
            os.makedirs(os.path.join(FLAGS.logs_dir))
    except OSError as e:
        # if e.errno != errno.EEXIST:
        print(e.errno)
        print("Failed to create directory!!!!!")
        raise
    ###########################

    with open(os.path.join(FLAGS.logs_dir, log_txt_name), 'a') as log_text:

        print('\n')

        print('==============================================================================')
        print('==============================================================================\r', file=log_text)
        if FLAGS.mode == 'train':
            print("Mode is set to 'train' (%s). \n" % datetime.datetime.now())
            print("Mode is set to 'train' (%s). \r" % datetime.datetime.now(), file=log_text)
        elif FLAGS.mode == 'visualize':
            log_text.write("Mode is set to 'visualize' (%s)." % datetime.datetime.now())
            print("Mode is set to 'visualize' (%s).\n" % datetime.datetime.now())
            print("Mode is set to 'visualize' (%s). \r" % datetime.datetime.now(), file=log_text)

        # Show and save parameters
        print('##Parameters##')
        print('GPU: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
        print('start_fold: ' + str(start_fold))
        print('learning_rate: '+str(FLAGS.learning_rate))
        print('Optimizer: '+ str(name_of_optimizer))
        print('The number of epochs: '+str(num_of_epochs))
        print('The number of classes: ' + str(num_of_classes))
        print('drop out rate: ' + str(drop_out_value))
        print('loss_name: ' + str(loss_name))
        print('train_batch_size: ' + str(FLAGS.training_batch_size))
        print('learning_rate: ' + str(FLAGS.learning_rate))

        if resize_option['resize']==False:
            print('resize option:False')
        elif resize_option['resize']==True:
            print('resize option: True')
            print('resize_size: '+ str(resize_size))
        print('\n')

        # Show and save parameters save
        print('##Parameters##\r', file=log_text)
        print('GPU: ' + str(os.environ['CUDA_VISIBLE_DEVICES'])+'\r', file=log_text)
        print('train_batch_size: ' + str(FLAGS.training_batch_size)+'\r', file=log_text)
        print('learning_rate: ' + str(FLAGS.learning_rate)+'\r', file=log_text)
        print('optimizer: ' + str(name_of_optimizer)+'\r', file=log_text)
        print('The number of epochs: ' + str(num_of_epochs)+'\r', file=log_text)
        print('The number of classes: ' + str(num_of_classes)+'\r', file=log_text)
        print('drop out rate: ' + str(drop_out_value)+'\r', file=log_text)
        print('loss_name: ' + str(loss_name) + '\r', file=log_text)
        if resize_option['resize'] == False:
            print('resize option:False\r', file=log_text)
        elif resize_option['resize'] == True:
            print('resize option: True\r', file=log_text)
            print('resize_size: ' + str(resize_size)+'\r', file=log_text)
        print('\r', file=log_text)


        print('1. Making and calling image list...')
        print('1. Making and calling image list... \r', file=log_text)

        # pickle_records = Make_cv_pickle_List.make_or_load_list_of_pickle(FLAGS.data_dir)
        # print(len(pickle_records))
        # print('Pickle is Ready!!!!!!!!!!')
        # pause()

        # ==============================================================================================
        #   patch 위치 별로 nonzero ground truth 리스트로 만들어준 .pickle calling
        # ==============================================================================================

        Sub_add_Name_list = []
        for Par_dir, Sub_Name, File_name in os.walk(FLAGS.data_dir):
            if Sub_Name:
                for itr in range(len(Sub_Name)):
                    if 'Osteolytic' in Sub_Name[itr].split('_'):
                        Sub_add_Name_list.append(Sub_Name[itr])
        train_File_add_path_list = []
        for i in range(len(Sub_add_Name_list)):
            train_add_path = os.path.join(FLAGS.data_dir,Sub_add_Name_list[i])
            train_list = os.listdir(train_add_path)
            train_File_add_path_list.append(train_list)
        train_File_add_path_list = sum(train_File_add_path_list, [])
        train_File_add_path_list.sort(key=natural_keys)

        Sub_Name_list = []
        for Par_dir, Sub_Name, File_name in os.walk(FLAGS.data_dir):
            if Sub_Name:
                for itr in range(len(Sub_Name)):
                    if 'v0.2' in Sub_Name[itr].split('_'):
                        Sub_Name_list.append(Sub_Name[itr])
        train_File_path_list = []
        for i in range(len(Sub_Name_list)):
            train_path = os.path.join(FLAGS.data_dir,Sub_Name_list[i])
            train_list = os.listdir(train_path)
            train_File_path_list.append(train_list)
        train_File_path_list = sum(train_File_path_list, [])

        fold_1 = []
        fold_2 = []
        fold_3 = []
        fold_4 = []
        fold_5 = []
        for itr2 in range(len(train_File_path_list)):
            if [words for words in ['1'] if words in train_File_path_list[itr2].split('_')]:
                fold_1.append(train_File_path_list[itr2])
            elif [words for words in ['2'] if words in train_File_path_list[itr2].split('_')]:
                fold_2.append(train_File_path_list[itr2])
            elif [words for words in ['3'] if words in train_File_path_list[itr2].split('_')]:
                fold_3.append(train_File_path_list[itr2])
            elif [words for words in ['4'] if words in train_File_path_list[itr2].split('_')]:
                fold_4.append(train_File_path_list[itr2])
            elif [words for words in ['5'] if words in train_File_path_list[itr2].split('_')]:
                fold_5.append(train_File_path_list[itr2])
            else:
                pass

        train_numpy_list = [fold_1, fold_2, fold_3, fold_4, fold_5]

        valid_File_path_list = []
        for Par_dir, Sub_Name, File_name in os.walk(FLAGS.data_dir):
            if File_name:
                for itr in range(len(File_name)):
                    if 'All' in File_name[itr].split('_'):
                        valid_File_path_list.append(File_name[itr])
        valid_File_path_list.sort(key=natural_keys)
        # ==============================================================================================
        pickle_list = list(zip(train_numpy_list, valid_File_path_list))

        # 설정한 하나의 gpu에서 5fold가 차례대로 돌아갈 수 있도록
        # #################################################################
        # #################### 5 Fold Cross Validation ####################
        # #################################################################
        # start_fold = 0
        # kf = KFold(n_splits=5)
        # for train_index, valid_index in kf.split(pickle_list):
        #     start_fold += 1
        #     print('\r')
        #     print('#######################################################')
        #     print('Now starting Cross Validation Fold: %d' % (start_fold))
        #     print('#######################################################')
        #     print('\r')
        #     # print("Train:", train_index, "Valid:", valid_index)
        #     # print("Train:", len(train_index), "Valid:", len(valid_index))
        #     # pause()
        #     print('\r')
        #     print('#######################################################')
        #     print('Cross Validation Train Num: %d' % len(train_index))
        #     print('Cross Validation Valid Num: %d' % len(valid_index))
        #     print('#######################################################')
        #     print('\r')
        #
        #     train_patient_records = []
        #     for i in train_index:
        #         train = pickle_list[i][0]
        #         train_patient_records.append(train)
        #
        #     valid_patient_records = []
        #     for j in valid_index:
        #         valid = pickle_list[j][1]
        #         valid_patient_records.append(valid)

        if start_fold == 1:
            train_patient_records = [pickle_list[1][0], pickle_list[2][0], pickle_list[3][0], pickle_list[4][0]]
            valid_patient_records = [pickle_list[0][1]]
        elif start_fold == 2:
            train_patient_records = [pickle_list[0][0], pickle_list[2][0], pickle_list[3][0], pickle_list[4][0]]
            valid_patient_records = [pickle_list[1][1]]
        elif start_fold == 3:
            train_patient_records = [pickle_list[0][0], pickle_list[1][0], pickle_list[3][0], pickle_list[4][0]]
            valid_patient_records = [pickle_list[2][1]]
        elif start_fold == 4:
            train_patient_records = [pickle_list[0][0], pickle_list[1][0], pickle_list[2][0], pickle_list[4][0]]
            valid_patient_records = [pickle_list[3][1]]
        elif start_fold == 5:
            train_patient_records = [pickle_list[0][0], pickle_list[1][0], pickle_list[2][0], pickle_list[3][0]]
            valid_patient_records = [pickle_list[4][1]]

        train_patient_records = sum(train_patient_records, [])
        # print('train_patient_records',train_patient_records)
        # print('valid_patient_records',valid_patient_records)

        train_patient_add_records = []
        for itr2 in range(len(train_File_add_path_list)):
            fold_num = re.split('[d.]', train_File_add_path_list[itr2])[-2]
            if int(fold_num) == int(start_fold):
                train_patient_add_records.append(train_File_add_path_list[itr2])
        print(train_patient_records)
        print(train_patient_add_records)
        pause()

        #################################################################
        ###################  Fold Data set Calling.. ####################
        #################################################################
        add_pickle_file_path = os.path.join(train_add_path, train_patient_add_records[0])
        with open(add_pickle_file_path, 'rb') as f:
            train_add_records = pickle.load(f)
        print("Calling additional data considering HU...")
        print(train_add_records[0:3])
        print('\r')
        pre_train_records = Make_cv_pickle_List.load_pickle(train_path, train_patient_records)
        valid_records = Make_cv_pickle_List.load_pickle(FLAGS.data_dir, valid_patient_records)
        train_records = pre_train_records + train_add_records

        print('\r')
        print('#######################################################\r')
        print('Now starting Cross Validation Fold: %d' % (start_fold))
        print('#######################################################\r')
        print('len(train_records)',len(pre_train_records), '        : Original')
        print('len(train_add_records)', len(train_add_records), '     : Added')
        print('------------------------------------------------------')
        print('Total Training data Number:',len(train_records))
        print('Total Validation data Number:', len(valid_records))
        print('#######################################################\r')
        print('\r')
        pause()

        print('===original_train_records type and shape===')
        print('===original_train_records type and shape===\r', file=log_text)
        print(type(train_records))
        print(str(type(train_records))+'\r', file=log_text)
        print(len(train_records))
        print(str(len(train_records))+'\r', file=log_text)
        print('\r', file=log_text)

        print('===original_train_records[0~%d]' % FLAGS.training_batch_size + ' values===')
        print('===original_train_records[0~%d]' % FLAGS.training_batch_size + ' values===\r', file=log_text)
        for itr in range(FLAGS.training_batch_size):
            print(train_records[itr])
            print(str(train_records[itr]) + '\r', file=log_text)
        print('\r', file=log_text)

        print('===valid_records type and shape===')
        print('===valid_records type and shape===\r', file=log_text)
        print(type(valid_records))
        print(str(type(valid_records))+'\r',file=log_text)
        print(len(valid_records))
        print(str(len(valid_records))+'\r',file=log_text)
        print('\r', file=log_text)

        print('===valid_records[0~%d]' % FLAGS.validation_batch_size + ' values===')
        print('===valid_records[0~%d]' % FLAGS.validation_batch_size + ' values===\r', file=log_text)
        for itr in range(FLAGS.validation_batch_size):
            print(valid_records[itr])
            print(str(valid_records[itr])+'\r', file=log_text)
        print('\r', file=log_text)


        #################################################################
        #################epoch and iteration calculation#################
        #################################################################
        no_total_train_records = np.array(len(train_records)).astype(np.int)
        print('\r')
        print('#######################################################')
        print('The number of total train records: %d' % (no_total_train_records))
        print('#######################################################')
        print('\r')

        if augmentation_option['augmentation']:
            if LR_reverse_option['LR_reverse']:
                no_total_train_records *= len(LR_reverse_option['LR_case'])
            else:
                pass
            if add_to_pixel_option['add_to_pixel']:
                no_total_train_records *= len(add_to_pixel_option['add_value'])
            else:
                pass
        else:
            print('No augmentation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        print('\r')
        print('#######################################################')
        print('The number of total train records with Augmentation: %d' % (no_total_train_records))
        print('#######################################################')
        print('\r')

        print('\r', file=log_text)
        print('####################################################### \r', file=log_text)
        print('The number of total train records with augmentation: %d \r' % no_total_train_records, file=log_text)
        print('####################################################### \r', file=log_text)

        itr_per_training_epoch = int(no_total_train_records / FLAGS.training_batch_size)
        print('The number of iteration per epoch of total training data: %d' % itr_per_training_epoch)
        print('The number of iteration per epoch of total training data: %d \r' % itr_per_training_epoch, file=log_text)

        MAX_ITERATION = int(itr_per_training_epoch * FLAGS.epochs)
        print('The number of MAX_ITERATION: %d \r' % MAX_ITERATION)
        print('The number of MAX_ITERATION: %d \r' % MAX_ITERATION, file=log_text)


        if len(valid_records) % FLAGS.validation_batch_size != 0:
            print('validation_batch_size is not suitable for the number of train images.')
            print('You should select batch_size one of the values below.')
            for j in range(1, len(valid_records)):
                if len(valid_records) % j == 0:
                    print(j, end=" ")
                if (j + 1) == len(valid_records):
                    print(j + 1, end=" ")
            return None
        else:
            itr_per_valid_epoch = int((len(valid_records)) / (FLAGS.validation_batch_size))
        ######################################################################


        print('2. Getting size of data...')
        print('2. Getting size of data...\r', file=log_text)
        sample_train_set_loader = Bone_Meta_Data_3cto3v_Loader_patch.data_loader(train_records, resize_option)
        batch_pre_training_image, batch_main_training_image, batch_post_training_image, batch_main_training_annotation = sample_train_set_loader.get_next_batch(2)

        sample_train_set_loader_valid = Bone_Meta_Data_3cto3v_Loader_512_size.data_loader(train_records, resize_option)
        batch_pre_valid_image, batch_main_valid_image, batch_post_valid_image, batch_main_valid_annotation = sample_train_set_loader_valid.get_next_batch(
            2)

        print('===== train set check =====')
        print('===== train set check =====\r', file=log_text)
        print('pre        : ', batch_pre_training_image.shape)
        print('main       : ', batch_main_training_image.shape)
        print('post       : ', batch_post_training_image.shape)
        print('annotation : ', batch_main_training_annotation.shape)
        print('===== valid set check =====')
        print('===== valid set check =====\r', file=log_text)
        print('pre        : ', batch_pre_valid_image.shape)
        print('main       : ', batch_main_valid_image.shape)
        print('post       : ', batch_post_valid_image.shape)
        print('annotation : ', batch_main_valid_annotation.shape)

        print(str(batch_pre_training_image.shape)+'\r', file=log_text)
        print(str(batch_main_training_image.shape)+'\r', file=log_text)
        print(str(batch_post_training_image.shape)+'\r', file=log_text)
        print(str(batch_main_training_annotation.shape)+'\r', file=log_text)
        print('\r', file=log_text)


        print('========<Notice>========')
        print('Test set will be loaded later...')
        print('========================')
        print('\n')

        # Get batch_training_image size
        shape_of_main_image = np.array(batch_main_training_image).shape
        image_height = shape_of_main_image[1]
        image_width = shape_of_main_image[2]
        image_channel = shape_of_main_image[3]

        # Get batch_training_annotation size
        shape_of_GroundTruth = np.array(batch_main_training_annotation).shape
        GroundTruth_height = shape_of_GroundTruth[1]
        GroundTruth_width = shape_of_GroundTruth[2]
        GroundTruth_channel = shape_of_GroundTruth[3]

        print('3. Placeholder Declaration...')
        print('3. Placeholder Declaration...\r', file=log_text)
        pre_input_image = tf.placeholder(tf.float32, shape=[None, None, None, image_channel], name='pre_input_image')
        main_input_image = tf.placeholder(tf.float32, shape=[None, None, None, image_channel], name='main_input_image')
        post_input_image = tf.placeholder(tf.float32, shape=[None, None, None, image_channel], name='post_input_image')
        ground_truth = tf.placeholder(tf.int32, shape=[None, None, None, GroundTruth_channel], name='ground_truth')

        keep_probability = tf.placeholder(tf.float32, name='keep_probability')

        print('4. Loading VGG-19 weights...')
        print('4. Loading VGG-19 weights...\r', file=log_text)
        weight_set, normal_set = Weights_Loader.weights_loader(FLAGS.model_dir, FLAGS.model_file_name)

        print('5. Loading Inference module...')
        print('5. Loading Inference module...\r', file=log_text)
        pred_annotation, logits = Inference.inference_process(pre_input_image, main_input_image, post_input_image, weight_set, normal_set, keep_probability, num_of_classes, FLAGS.debug_mode)

        tf.summary.image("pre_input_image", pre_input_image, max_outputs=2)
        tf.summary.image("main_input_image", main_input_image, max_outputs=2)
        tf.summary.image("post_input_image", post_input_image, max_outputs=2)
        tf.summary.image('ground_truth', tf.cast(ground_truth, tf.uint8), max_outputs=2)
        tf.summary.image('pred_annotation', tf.cast(pred_annotation, tf.uint8), max_outputs=2)

        print('6. Defining loss function and trainer...')
        print('6. Defining loss function and trainer...\r', file=log_text)

        # loss = cost_sensitive_loss.weighted_loss(logits, ground_truth, num_of_classes, GroundTruth_height, GroundTruth_width, GroundTruth_channel)
        # weighted loss for cost sensitive learning.

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(ground_truth, squeeze_dims=[3]), name='entropy'))
        tf.summary.scalar('entropy', loss)
        trainable_var = tf.trainable_variables()

        if FLAGS.debug_mode:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)

        train_operator = opti_trainer.train(loss, trainable_var, FLAGS.learning_rate, FLAGS.debug_mode)

        print("7. Setting up summary operator...")
        print("7. Setting up summary operator...\r", file=log_text)
        summary_operator = tf.summary.merge_all()  # 이 함수도 확인하기

        print('8. Opening Session...')
        print('8. Opening Session...\r', file=log_text)
        sess_start = datetime.datetime.now()
        sess = tf.Session()

        print('9. Setting up Saver...')
        print('9. Setting up Saver...\r', file=log_text)
        saver = tf.train.Saver()  # 이거도 보자
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)  # tf.summary 뜯어봐야 함.

        print('10. Initializing global variables...')
        print('10. Initializing global variables...\r', file=log_text)
        init = tf.global_variables_initializer()
        sess.run(init)
        print('\n')
        print('\r', file=log_text)

        '''
        ========================
        ckpt model calling
        ========================
        '''
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)  # 이 함수 확인하기
        if ckpt and ckpt.model_checkpoint_path:  # 두 개가 같다고 하면 ==의 의미가 if ... and...
            print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored...')
            print('Model restored...\r', file=log_text)


        """
        ===================================
       ｜                                  ｜
       ｜  Train/Visualize/Test mode Part  ｜
       ｜                                  ｜
        ===================================
        """
        if FLAGS.mode == 'train': # train and validation together

            print('11. FLAGS.mode is set to "train"')
            print('12. Setting up train/valid batch loader...')

            print('11. FLAGS.mode is set to "train"\r', file=log_text)
            print('12. Setting up train/valid batch loader...\r', file=log_text)

            random.shuffle(train_records)
            train_batch_loader = Bone_Meta_Data_3cto3v_Loader_patch.data_loader(train_records, resize_option)

            print('train_batch_loader ready!')
            print('train_batch_loader ready!\r', file=log_text)

            # random.shuffle(valid_records)
            valid_batch_loader = Bone_Meta_Data_3cto3v_Loader_512_size.data_loader(valid_records, resize_option)

            print('valid_batch_loader ready!')
            print('valid_batch_loader ready!\r', file=log_text)
            print('\n')
            print('\r', file=log_text)

            ''''''
            valid_step_count = 0
            train_epoch_count = 0
            valid_epoch_count = 0
            zero_patch_ground_truth = 0

            for itr in range(MAX_ITERATION):
                train_batch_pre_images, train_batch_main_images, train_batch_post_images, train_batch_main_annotation = train_batch_loader.get_next_batch(
                    FLAGS.training_batch_size)

                print(np.array(train_batch_main_images).shape)
                if np.count_nonzero(train_batch_main_annotation[0]) == 0:
                    zero_patch_ground_truth += 1
                    print('>>>zero_patch_ground_truth:', zero_patch_ground_truth)
                    raise ValueError
                else:
                    pass

                '''
                =======================
                2-2. Augmentation
                =======================
                '''

                if augmentation_option['augmentation']==True:

                    if len(train_batch_pre_images) != len(train_batch_main_images) != len(train_batch_post_images) != len(train_batch_main_annotation):
                        print('Length of batch images does not match. Please check!')

                        raise ValueError
                    else:
                        pass

                    for iteration in range(len(train_batch_main_images)):
                        pre_img_temp = train_batch_pre_images[iteration]
                        main_img_temp = train_batch_main_images[iteration]
                        post_img_temp = train_batch_post_images[iteration]
                        annot_img_temp = train_batch_main_annotation[iteration]

                        train_batch_pre_images_aug = []
                        train_batch_main_images_aug =[]
                        train_batch_post_images_aug = []
                        train_batch_main_annotation_aug = []

                        if LR_reverse_option['LR_reverse']:
                            on_off_alert = random.choice(LR_reverse_option['LR_case'])

                            if on_off_alert == 1:
                                pre_img_temp = np.fliplr(pre_img_temp)
                                main_img_temp = np.fliplr(main_img_temp)
                                post_img_temp = np.fliplr(post_img_temp)
                                annot_img_temp = np.fliplr(annot_img_temp)
                            else:
                                pass

                        if add_to_pixel_option['add_to_pixel']:
                            on_off_alert2 = random.choice(add_to_pixel_option['add_value'])

                            pre_img_temp = (pre_img_temp + on_off_alert2)
                            main_img_temp = (main_img_temp + on_off_alert2)
                            post_img_temp = (post_img_temp + on_off_alert2)

                            train_batch_pre_images_aug.append(pre_img_temp)
                            train_batch_main_images_aug.append(main_img_temp)
                            train_batch_post_images_aug.append(post_img_temp)
                            train_batch_main_annotation_aug.append(annot_img_temp)

                feed_dict = {pre_input_image: train_batch_pre_images_aug, main_input_image: train_batch_main_images_aug, post_input_image: train_batch_post_images_aug, ground_truth: train_batch_main_annotation_aug, keep_probability: drop_out_value}

                sess.run(train_operator, feed_dict=feed_dict)

                if (itr + 1) % (round(itr_per_training_epoch / itr_per_training_epoch)) == 0:
                    train_loss, summary_str = sess.run([loss, summary_operator], feed_dict=feed_dict)
                    if (itr + 1) % 10 == 0:
                        print('     Training step: %d, train_loss: %f' % (itr + 1, train_loss),
                              ' l Time:', datetime.datetime.now() - sess_start, 'l')
                    if (itr + 1) % 100 == 0:
                        print('     Training step: %d, train_loss: %f\r' % (itr + 1, train_loss), file=log_text)
                    summary_writer.add_summary(summary_str, itr)
                    ###########################
                else:
                    pass

                ######Validation Step######
                if itr > 0:
                    if (itr + 1) % itr_per_training_epoch == 0:
                        valid_step_count += 1
                        train_epoch_count += 1
                        print("************************" + str(
                            train_epoch_count) + ' train epoch completed' + "************************* \r",
                              file=log_text)
                        print("************************" + str(
                            train_epoch_count) + ' train data epoch completed' + "************************* \r")

                        saver.save(sess, FLAGS.logs_dir + "/model.ckpt", global_step=(itr + 1))
                        print('model saved \r')
                        print('model saved \r', file=log_text)

                        valid_epoch_count += 1
                        batch_number_jump = 0

                        for valid_total_data_itr in range(itr_per_valid_epoch):
                            valid_pre_batch_images_for_save, valid_main_batch_images_for_save, valid_post_batch_images_for_save, valid_main_batch_annotation_for_save = valid_batch_loader.get_next_batch(
                                FLAGS.validation_batch_size)

                            valid_step_feed_dict = {pre_input_image: valid_pre_batch_images_for_save, main_input_image: valid_main_batch_images_for_save, post_input_image: valid_post_batch_images_for_save, ground_truth: valid_main_batch_annotation_for_save, keep_probability: 1.0}

                            valid_loss, valid_pred = sess.run([loss, pred_annotation], feed_dict=valid_step_feed_dict)

                            print("%d validation step:  %s ---> Validation_loss: %f" % (
                            (valid_total_data_itr + 1), datetime.datetime.now(), valid_loss))
                            print("%d validation step:  %s ---> Validation_loss: %f \r" % (
                            (valid_total_data_itr + 1), datetime.datetime.now(), valid_loss),
                                  file=log_text)
                            print('\r', file=log_text)

                            valid_main_batch_images_for_save = np.squeeze(valid_main_batch_images_for_save, axis=-1)
                            valid_batch_main_annotation = np.squeeze(valid_main_batch_annotation_for_save, axis=-1)
                            valid_pred = np.squeeze(valid_pred, axis=-1)

                            print('FLAGS.validation_batch_size:',FLAGS.validation_batch_size)

                            for valid_saving_itr in range(FLAGS.validation_batch_size):
                                utils.save_image(valid_main_batch_images_for_save[valid_saving_itr], FLAGS.logs_dir,
                                                 name='valid_step_input_image_' + str(start_fold)+ 'fold_' + str(valid_step_count) + '_' + str(
                                                     valid_saving_itr + 1 + batch_number_jump))

                                utils.save_image(valid_batch_main_annotation[valid_saving_itr], FLAGS.logs_dir, name='valid_step_ground_truth_' + str(start_fold)+ 'fold_' + str(valid_step_count) + '_' + str(valid_saving_itr + 1 + batch_number_jump))

                                utils.save_image(valid_pred[valid_saving_itr], FLAGS.logs_dir,
                                                 name='valid_step_pred_annot_' + str(start_fold)+ 'fold_' + str(valid_step_count) + '_' + str(
                                                     valid_saving_itr + 1 + batch_number_jump))
                                print("Saved image: %d" % (valid_saving_itr + 1 + batch_number_jump))
                                print("Saved image: %d\r" % (valid_saving_itr + 1 + batch_number_jump), file=log_text)

                            batch_number_jump = valid_saving_itr + 1 + batch_number_jump
                            print('\r', file=log_text)
                            print('\n')

            tf.reset_default_graph()
            print("This training took for ", datetime.datetime.now() - sess_start)

                            # ==========================#



        elif FLAGS.mode == 'visualize':
            print('11. FLAGS.mode is set to "visualize"')
            print('12. Setting up visualize valid batch loader...')

            print('11. FLAGS.mode is set to "visualize"\r', file=log_text)
            print('12. Setting up visualize valid batch loader...\r', file=log_text)
            print('\r', file=log_text)

            # np.random.shuffle(valid_records)
            vis_mode_valid_batch_loader = Bone_Meta_Data_3cto3v_Loader_512_size.data_loader(valid_records, resize_option)

            print('vis_mode_valid_batch_loader ready!')
            print('vis_mode_valid_batch_loader ready!\r', file=log_text)
            vis_mode_valid_batch_pre_image, vis_mode_valid_batch_main_image, vis_mode_valid_batch_post_image, vis_mode_valid_batch_main_annotation = vis_mode_valid_batch_loader.get_next_batch(
                FLAGS.batch_size)

            pred = sess.run(pred_annotation, feed_dict={pre_input_image: vis_mode_valid_batch_pre_image, main_input_image: vis_mode_valid_batch_main_image, post_input_image: vis_mode_valid_batch_post_image, ground_truth: vis_mode_valid_batch_main_annotation, keep_probability: 1.0})

            vis_mode_valid_batch_annot = np.squeeze(vis_mode_valid_batch_main_annotation, axis=3)
            pred = np.squeeze(pred, axis=3)

            for itr in range(FLAGS.batch_size):
                utils.save_image(vis_mode_valid_batch_main_image[itr], FLAGS.logs_dir, name='vis_mode_input_' + str(itr))
                utils.save_image(vis_mode_valid_batch_annot[itr], FLAGS.logs_dir, name="vis_mode_ground_truth_" + str(itr))
                utils.save_image(pred[itr], FLAGS.logs_dir, name="vis_mode_pred_" + str(itr))
                print("Saved image: %d" % itr)
                print("Saved image: %d\r" % itr, file=log_text)

        '''
            아래는 test mode
        '''
        # elif FLAGS.mode == 'test':
        # test_dataset_reader = dataset.BatchDatset(valid_records, image_options) #왜 valid data 가지고 하는지 모르겠지만..
        #     for itr in range(len(valid_records)):
        #         test_imagesAd, test_annotationsAd, test_annotationsIdxAd = test_dataset_reader.next_batch(
        #             FLAGS.batch_size)
        #         test_images = []
        #         test_annotations = []
        #         for i in xrange(FLAGS.batch_size):
        #             test_images.append(test_dataset_reader._transformImg(test_imagesAd[i]))
        #             test_annotations.append(
        #                 test_dataset_reader._transformAnnot(test_annotationsAd[i], test_annotationsIdxAd[i]))
        #
        #         test_images = np.resize(np.array(test_images), [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])
        #         test_annotations = np.resize(np.array(test_annotations), [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])
        #
        #         pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations,
        #                                                     keep_probability: 1.0})
        #
        #         for i in range(FLAGS.batch_size):
        #             print(confusion_matrix(np.reshape(test_annotations[i], -1), np.reshape(pred[i], -1)))
        #         itr = + FLAGS.batch_size

if __name__ == '__main__':
    tf.app.run()
