"""
#---------------------------*[Description]*---------------------------------#
#            * Make pickle file for cross validation training
#---------------------------------------------------------------------------#
#     Date         |   Name     |           Version description             #
#---------------------------------------------------------------------------#
# 2019.03.06                          5 fold cross validation pickle file
                                             training 200*200
#---------------------------------------------------------------------------#
"""
import os, glob
import numpy as np
import nibabel
import tensorflow as tf
import pickle
from scipy.ndimage import label
import matplotlib.pyplot as plt
import pydicom

def pause():
    input('Press the <Enter> key to continue...')

def load_pickle_numpy(Par_data_dir, pickle_name):
    pre_image_train = []
    main_image_train = []
    post_image_train = []
    main_annot_train = []
    pickle_number = 0
    for i in range(len(pickle_name)):
        pickle_number += 1
        pickle_file_path = os.path.join(Par_data_dir, pickle_name[i])
        print(pickle_number,'/520: ', pickle_file_path)
        with open(pickle_file_path, 'rb') as f:
            result = pickle.load(f)
            pre_image = result['pre_image']
            main_image = result['main_image']
            post_image = result['post_image']
            main_annot = result['main_annot']

            pre_image_train.append(pre_image)
            main_image_train.append(main_image)
            post_image_train.append(post_image)
            main_annot_train.append(main_annot)

            # print('len(main_image):', len(main_image))
            # print('np.array(main_image).shape:', np.array(main_image).shape)

    return pre_image_train, main_image_train, post_image_train, main_annot_train

def load_pickle(Par_data_dir, pickle_name):
    file_list_valid = []
    for i in range(len(pickle_name)):
        pickle_file_path = os.path.join(Par_data_dir, pickle_name[i])
        with open(pickle_file_path, 'rb') as f:
            result = pickle.load(f)
            result = list(result.values())
            file_list_valid.append(result)

    file_list_valid = sum(file_list_valid, [])
    file_list_valid = sum(file_list_valid, [])

    return file_list_valid


def make_or_load_list_of_pickle(Par_data_dir):

    pickle_name = 'BM_CV_fold_1_ROI_spine_crop(-25, 75)_list.pickle'
    # pickle_name = 'BM_CV_fold_2_ROI_spine_crop(-25, 75)_list.pickle'
    # pickle_name = 'BM_CV_fold_3_ROI_spine_crop(-25, 75)_list.pickle'
    # pickle_name = 'BM_CV_fold_4_ROI_spine_crop(-25, 75)_list.pickle'
    # pickle_name = 'BM_CV_fold_5_ROI_spine_crop(-25, 75)_list.pickle'


    # pickle_name = 'BM_CV_fold_1_All_spine.pickle'
    # pickle_name = 'BM_CV_fold_2_All_spine.pickle'
    # pickle_name = 'BM_CV_fold_3_All_spine.pickle'
    # pickle_name = 'BM_CV_fold_4_All_spine.pickle'
    # pickle_name = 'BM_CV_fold_5_All_spine.pickle'
    print(pickle_name)

    pickle_file_path = os.path.join(Par_data_dir, pickle_name)

    if not os.path.exists(pickle_file_path):
        print(print('There is no pickle file. \n'))
        print('We are in the data listing step. \n')

        list_of_data = create_data_list(os.path.join(Par_data_dir))

        print("Pickling ...")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(list_of_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Pickle file already exists! I will use this .pickle file \n')

    with open(pickle_file_path, 'rb') as f:
        result = pickle.load(f)
        pickle_records = result['fold_1']
        del result

    return pickle_records


def create_data_list(Par_dir):
    if not tf.gfile.Exists(Par_dir):
        print("Could not find the 'input' directory. Set again. \n")
        return None

    sub_par_name_in_Par_dir = ['fold_1']
    sub_par_directory_path_list = []
    img_annot_list = {}

    for sub_par_name in sub_par_name_in_Par_dir:
        Par_dir_list = []
        FilePath_list = []
        FileName_list = []

        for ParName, SubName, FileName in os.walk(os.path.join(Par_dir, sub_par_name)):
            if FileName != []:
                Par_dir_list.append(ParName)

                for filename in FileName:
                    FileName_list.append(os.path.join(ParName, filename))
        # print('Par_dir_list',Par_dir_list)
        bundle_record = []
        for folder_name in Par_dir_list:
            # print('folder_name:',folder_name)
            nii_file_list = glob.glob(os.path.join(folder_name, '*.gz'))
            if not nii_file_list:
                nii_file_list = glob.glob(os.path.join(folder_name, '*.nii'))

            dcm_file_list = glob.glob(os.path.join(folder_name, '*.dcm'))
            dcm_file_list.sort(reverse=True)

            # nii indexing step
            nii_file = nibabel.load(nii_file_list[0])
            np_array_nii_file = nii_file.get_data()
            no_of_nii_images = np_array_nii_file.shape[2] # index 할 image 수

            # ground_truth_stack = []
            # for index in range(no_of_nii_images):
            #     ground_truth_array = np_array_nii_file[:][:, :][:, :, index]
            #     ground_truth_stack.append(ground_truth_array)
            #
            # gt_labeled_array, gt_num_features = label(ground_truth_stack)
            # gt_unique, gt_counts = np.unique(gt_labeled_array, return_counts=True)
            # zip_label = list(zip(gt_unique, gt_counts))
            # print('======== Before Threshold ========')
            # print('th_gt_labeled_array:', gt_labeled_array.shape)
            # print('Original Meta Number:',len(gt_unique))
            #
            # label_value = [0]  # Including Background value
            # for itr in range(len(zip_label)):
            #     if zip_label[itr][1] < 150:  # 150 pixel threshold
            #         label_value.append(zip_label[itr][0])
            #
            # for z in range(len(gt_labeled_array)):
            #     for x in range(len(gt_labeled_array[z])):
            #         for y in range(len(gt_labeled_array[z])):
            #
            #             if gt_labeled_array[z][x, y] in label_value:
            #                 gt_labeled_array[z][x, y] = 0
            #             else:
            #                 gt_labeled_array[z][x, y] = 1
            #
            # gt_labeled_array = np.transpose(gt_labeled_array, (1, 2, 0))
            # th_gt_labeled_array, th_gt_num_features = label(gt_labeled_array)
            # th_gt_unique, th_gt_counts = np.unique(th_gt_labeled_array, return_counts=True)
            #
            # print('======== After Threshold ========')
            # print('======== (Th Value: 150) ========')
            # print('th_gt_labeled_array:', th_gt_labeled_array.shape)
            # print(np.asarray((th_gt_unique, th_gt_counts)).T)
            # print('=================================')
            # print('\n')

            # new_gt_labeled_array = gt_labeled_array
            # no_of_nii_images = new_gt_labeled_array.shape[2]  # index 할 image 수

            for index in range(no_of_nii_images):
                if index >= 50 :

                    record_list_temp = []

                    # For valid pickling check!
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[156:356, 206:406, index]))) != 0: # (0, 50)    /startx, starty: 156 206
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[156:356, 156:356, index]))) != 0: # (0, 0)     / startx, starty: 156 156
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[156:356, 256:456, index]))) != 0: # (0, 100)   /startx, starty: 156 256
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[106:306, 206:406, index]))) != 0: # (-50, 50)  / startx, starty: 106 206
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[206:406, 206:406, index]))) != 0: # (50, 50)   / startx, starty: 206 206
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[206:406, 156:356, index]))) != 0: # (50, 0)    / startx, starty: 206 156
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[206:406, 256:456, index]))) != 0: # (50, 100)  / startx, starty: 206 256
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[106:306, 156:356, index]))) != 0: # (-50, 0)   / startx, starty: 106 156
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[106:306, 256:456, index]))) != 0: # (-50, 100) / startx, starty: 106 256
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[181:381, 181:381, index]))) != 0: # (25, 25)   / startx, starty: 181 181
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[181:381, 231:431, index]))) != 0: # (25, 75)   / startx, starty: 181 231
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[131:331, 181:381, index]))) != 0: # (-25, 25)  / startx, starty: 131 181
                    # if np.count_nonzero(np.flipud(np.rot90(np_array_nii_file[131:331, 231:431, index]))) != 0: # (-25, 75)  / startx, starty: 131 231

                        # print(dcm_file_list[index])
                        # print('crop_image:',np.count_nonzero(np_array_nii_file[206:406, 156:356, index]))
                        # print('original_image:',np.count_nonzero(np_array_nii_file[:][:, :][:, :, index]))
                        # if dcm_file_list[index] == '/data/Data_Set/BM_CV/fold_2/_102029__0912120243/ser005img00123.dcm':
                        #     pause()
                        #
                        #     flip_fig = plt.figure('before flip')
                        #     sub_before = flip_fig.add_subplot(1, 5, 1) #크롭한 원하는 모습
                        #     sub_before.imshow(np.flipud(np.rot90(np_array_nii_file[156:356, 206:406, index])), cmap=plt.cm.bone)
                        #     sub_before.axis('off')
                        #     sub_after = flip_fig.add_subplot(1, 5, 2)  #그라운드 트루스 내가 보고싶은 모습
                        #     sub_after.imshow(np.flipud(np.rot90(np_array_nii_file[:][:, :][:, :, index])), cmap=plt.cm.bone)
                        #     sub_after.axis('off')
                        #     sub_after = flip_fig.add_subplot(1, 5, 3)  #잘못한 패치
                        #     sub_after.imshow(np_array_nii_file[156:356, 206:406, index], cmap=plt.cm.bone)
                        #     sub_after.axis('off')
                        #     sub_after = flip_fig.add_subplot(1, 5, 4)  #잘못한 패치
                        #     sub_after.imshow(np_array_nii_file[206:406, 156:356, index], cmap=plt.cm.bone)
                        #     sub_after.axis('off')
                        #     sub_after = flip_fig.add_subplot(1, 5, 5)  #잘못한 그라운드 트루스
                        #     sub_after.imshow(np_array_nii_file[:][:, :][:, :, index], cmap=plt.cm.bone)
                        #     sub_after.axis('off')
                        #     flip_fig.show()
                        #     pause()
                    if np.count_nonzero(np_array_nii_file[:][:, :][:, :, index]) != 0:

                        #아래 코드는 False 부분 날리는 코드
                        crop_option = (-25, 75)
                        if index + 1 > no_of_nii_images - 1 or index - 1 < 0:
                            continue

                        record_pre = {'pre_image': dcm_file_list[index - 1], 'pre_annotation': nii_file_list[0],
                                      'pre_annotIndex': index - 1, 'crop_option': crop_option}
                        record_main = {'main_image': dcm_file_list[index], 'main_annotation': nii_file_list[0],
                                    'main_annotIndex': index, 'crop_option': crop_option}
                        record_post = {'post_image': dcm_file_list[index + 1], 'post_annotation': nii_file_list[0],
                                       'post_annotIndex': index + 1, 'crop_option': crop_option}

                        record_list_temp.append(record_pre)
                        record_list_temp.append(record_main)
                        record_list_temp.append(record_post)
                        bundle_record.append(record_list_temp)

        img_annot_list[sub_par_name] = bundle_record

    return img_annot_list

