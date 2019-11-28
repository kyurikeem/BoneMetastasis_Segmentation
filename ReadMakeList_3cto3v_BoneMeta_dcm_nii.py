"""
#===========================================================================#
#---------------------------*[Description]*---------------------------------#
#           * This is the main code for BM 3ch to 3 vgg model
#===========================================================================#
#------------------------*[Record of version]*------------------------------#
#---------------------------------------------------------------------------#
#     Date         |   Name     |           Version description             #
#---------------------------------------------------------------------------#
# 2018.02.15  For Cross Valid, func. divided with ReadMakeList_per_patient.py
#---------------------------------------------------------------------------#
"""
import os, glob
import numpy as np
import nibabel
import tensorflow as tf
import pickle

def pause():
    input('Press the <Enter> key to continue...')

def create_train_data_list(Par_dir, train_patient_records):
    if not tf.gfile.Exists(Par_dir):
        print("Could not find the 'input' directory. Set again. \n")
        return None

    img_annot_list = {}

    Par_dir_list = []
    FilePath_list = []
    FileName_list = []
    for i in range(len(train_patient_records)):

        for ParName, SubName, FileName in os.walk(os.path.join(train_patient_records[i])):
            if FileName != []:
                Par_dir_list.append(ParName)
                for filename in FileName:
                    FileName_list.append(os.path.join(ParName, filename))

    bundle_record = []
    for folder_name in Par_dir_list:
        # print('Par_dir_list:', Par_dir_list)
        # pause()
        nii_file_list = glob.glob(os.path.join(folder_name, '*.gz'))
        if not nii_file_list:
            nii_file_list = glob.glob(os.path.join(folder_name, '*.nii'))

        # print('nii_file_list:',nii_file_list)
        # pause()
        # print('nii_file_list[0]:', nii_file_list[0])
        # pause()

        dcm_file_list = glob.glob(os.path.join(folder_name, '*.dcm'))
        dcm_file_list.sort(reverse=True)
        # print('dcm_file_list:',dcm_file_list)
        # pause()
        # print('dcm_file_list:',dcm_file_list[0])
        # pause()

        # nii indexing step
        nii_file = nibabel.load(nii_file_list[0])
        np_array_nii_file = nii_file.get_data()
        no_of_nii_images = np_array_nii_file.shape[2]  # index 할 image 수


        for index in range(no_of_nii_images):  #0~120까지

            record_list_temp = []
            # print('index:', index)
            if np.count_nonzero(np_array_nii_file[:][:, :][:, :, index]) != 0:
                # print(np.unique(np_array_nii_file[:][:, :][:, :, index]))
                # pause()

                # 아래 코드는 False 부분 날리는 코드
                if index + 1 > no_of_nii_images - 1 or index - 1 < 0:
                    continue

                record_pre = {'pre_image': dcm_file_list[index - 1], 'pre_annotation': nii_file_list[0],
                              'pre_annotIndex': index - 1}
                record_main = {'main_image': dcm_file_list[index], 'main_annotation': nii_file_list[0],
                               'main_annotIndex': index}
                record_post = {'post_image': dcm_file_list[index + 1], 'post_annotation': nii_file_list[0],
                               'post_annotIndex': index + 1}

                record_list_temp.append(record_pre)
                record_list_temp.append(record_main)
                record_list_temp.append(record_post)
                # print(record_list_temp)
                # pause()
                bundle_record.append(record_list_temp)

            img_annot_list = bundle_record
    # print('img_annot_list:',img_annot_list)
    # pause()
    return img_annot_list


def create_valid_data_list(Par_dir, valid_patient_records):
    if not tf.gfile.Exists(Par_dir):
        print("Could not find the 'input' directory. Set again. \n")
        return None

    img_annot_list = {}

    Par_dir_list = []
    FilePath_list = []
    FileName_list = []
    for i in range(len(valid_patient_records)):

        for ParName, SubName, FileName in os.walk(os.path.join(valid_patient_records[i])):
            if FileName != []:
                Par_dir_list.append(ParName)
                for filename in FileName:
                    FileName_list.append(os.path.join(ParName, filename))

    bundle_record = []
    for folder_name in Par_dir_list:
        nii_file_list = glob.glob(os.path.join(folder_name, '*.gz'))
        if not nii_file_list:
            nii_file_list = glob.glob(os.path.join(folder_name, '*.nii'))

        dcm_file_list = glob.glob(os.path.join(folder_name, '*.dcm'))
        dcm_file_list.sort(reverse=True)

        # nii indexing step
        nii_file = nibabel.load(nii_file_list[0])
        np_array_nii_file = nii_file.get_data()
        no_of_nii_images = np_array_nii_file.shape[2] # index 할 image 수

        for index in range(no_of_nii_images):

            record_list_temp = []
            # if np.count_nonzero(np_array_nii_file[:][:, :][:, :, index]) != 0:
                #아래 코드는 False 부분 날리는 코드
            if index + 1 > no_of_nii_images - 1 or index - 1 < 0:
                continue
            # print(index)
            # print(no_of_nii_images)

            record_pre = {'pre_image': dcm_file_list[index - 1], 'pre_annotation': nii_file_list[0],
                          'pre_annotIndex': index - 1}
            record_main = {'main_image': dcm_file_list[index], 'main_annotation': nii_file_list[0],
                        'main_annotIndex': index}
            record_post = {'post_image': dcm_file_list[index + 1], 'post_annotation': nii_file_list[0],
                           'post_annotIndex': index + 1}

            record_list_temp.append(record_pre)
            record_list_temp.append(record_main)
            record_list_temp.append(record_post)
            bundle_record.append(record_list_temp)

        img_annot_list = bundle_record

    return img_annot_list






