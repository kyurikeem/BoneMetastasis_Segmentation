"""
#---------------------------*[Description]*---------------------------------#
#            * Make pickle file for low HU value image data
#---------------------------------------------------------------------------#
"""
import numpy as np
import scipy.io as mat_file_loader
import imageio as image_loader
import nibabel as nib
import pydicom
import os
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt
# from scipy.misc import imsave

import image_crop

def pause():
    input('Press the <Enter> key to continue...')

class data_loader:

    batch_offset = 0
    epoch_count = 0

    def __init__(self, data_list, resize_option_dict):

        self.img_annot_list = data_list
        self.resize_option = resize_option_dict
        self.mat_file_in_dict = None
        self.filename = None
        self.specific_key = None
        self.np_array_values = None
        self.resized_image = None
        self.images = None
        self.annotations = None
        self.batch_file_list = None
        self.batch_image = None
        self.batch_main_annot = None
        self.pil_image = None
        self.squeezed_image = None
        self.original_size_image = None
        self.batch_image_temp = None
        self.batch_annot_temp = None
        self.temp_img = None
        self.temp_annot = None


    def file_type_check_and_load(self, one_of_list_values, index):

        if one_of_list_values.split('.')[-1] in ['jpg', 'png']:
            self.np_array_values = image_loader.imread(one_of_list_values)

        elif one_of_list_values.split('.')[-1] in ['dcm']:
            self.np_array_values = pydicom.dcmread(one_of_list_values).pixel_array

        elif one_of_list_values.split('.')[-1] == 'mat':
            self.mat_file_in_dict = mat_file_loader.loadmat(one_of_list_values)
            self.filename = os.path.splitext(one_of_list_values.split('/')[-1])[0] #Linux용
            # filename = os.path.splitext(one_of_list_values.split('\\')[-1])[0]  # Window용
            self.specific_key = self.filename + '_mask'  # this is only for meniere masking file
            self.np_array_values = self.mat_file_in_dict[self.specific_key] #이 부분 체크하기

        elif one_of_list_values.split('.')[-1] in ['nii', 'gz']:
            self.all_nii_file = nib.load(one_of_list_values)
            self.np_array_values_of_nii = self.all_nii_file.get_data()

            self.indexed_nii = self.np_array_values_of_nii[:][:, :][:, :, index]
            self.rotated_nii = np.rot90(self.indexed_nii)
            self.upside_down_nii = np.flipud(self.rotated_nii)
            self.np_array_values = self.upside_down_nii

        self.np_array_values = np.array(self.np_array_values)

        if len(self.np_array_values.shape) < 3:
            self.np_array_values = self.np_array_values[:, :, np.newaxis]

        return self.np_array_values

    def file_type_check_and_load_negative(self, one_of_list_values):

        if one_of_list_values.split('.')[-1] in ['jpg', 'png']:
            self.np_array_values = image_loader.imread(one_of_list_values)

        elif one_of_list_values.split('.')[-1] in ['dcm']:
            self.np_array_values = pydicom.dcmread(one_of_list_values).pixel_array

        elif one_of_list_values.split('.')[-1] == 'mat':
            self.mat_file_in_dict = mat_file_loader.loadmat(one_of_list_values)
            self.filename = os.path.splitext(one_of_list_values.split('/')[-1])[0] #Linux용
            self.specific_key = self.filename + '_mask'  # this is only for meniere masking file
            self.np_array_values = self.mat_file_in_dict[self.specific_key] #이 부분 체크하기

        self.np_array_values = np.array(self.np_array_values)

        if len(self.np_array_values.shape) < 3:
            self.np_array_values = self.np_array_values[:, :, np.newaxis]

        return self.np_array_values


    def size_transformer(self, different_size_images, resize_size):
        self.original_size_image = np.array(different_size_images)

        if self.original_size_image.shape[2] == 1:
            self.squeezed_image = np.squeeze(different_size_images)
            self.pil_image = Image.fromarray(self.squeezed_image)
            resized_image_temp = np.array(self.pil_image.resize((resize_size, resize_size), resample=Image.LANCZOS))
            self.resized_image = resized_image_temp[:, :, np.newaxis]
        else:
            self.pil_image = Image.fromarray(different_size_images)
            self.resized_image = self.pil_image.resize((resize_size, resize_size), resample=Image.LANCZOS)

        return self.resized_image


    def get_next_batch(self, batch_size):

        start = self.batch_offset
        self.batch_offset += batch_size

        if self.batch_offset > len(self.img_annot_list): # 여기서 batch_offset 값이 오버되면 batch_list shuffle

            # Display the number of epochs
            start = 0
            np.random.shuffle(self.img_annot_list)
            self.batch_offset = batch_size

            self.epoch_count += 1
            print("*****************" + str(self.epoch_count) + ' epoch completed' + "******************")

        end = self.batch_offset

        self.batch_file_list = self.img_annot_list[start:end]

        self.batch_image = []
        self.batch_pre_image = []
        self.batch_main_image = []
        self.batch_post_image = []
        self.batch_main_annot = []

        for i, file_list in enumerate(self.batch_file_list):

            pre_file_dict = file_list[0]
            main_file_dict = file_list[1]
            post_file_dict = file_list[2]

            # print('pre_file_dict:',pre_file_dict)
            # print('main_file_dict:',main_file_dict)
            # print('post_file_dict:',post_file_dict)

            if 'main_annotIndex' in main_file_dict:
                self.pre_img_temp = np.array(self.file_type_check_and_load(pre_file_dict['pre_image'], pre_file_dict['pre_annotIndex']))
                self.main_img_temp = np.array(self.file_type_check_and_load(main_file_dict['main_image'], main_file_dict['main_annotIndex']))
                self.post_img_temp = np.array(self.file_type_check_and_load(post_file_dict['post_image'], post_file_dict['post_annotIndex']))
                self.main_annot_temp = np.array(self.file_type_check_and_load(main_file_dict['main_annotation'], main_file_dict['main_annotIndex']))

                crop_option = main_file_dict['crop_option']
                # print(crop_option)

                self.pre_img_temp, self.main_img_temp, self.post_img_temp, self.main_annot_temp = image_crop.random_crop_center(self.pre_img_temp, self.main_img_temp, self.post_img_temp, self.main_annot_temp, 200, 200, crop_option)

                # if np.count_nonzero(self.main_annot_temp[:][:, :][:, :, 0]) != 0:

                self.batch_pre_image.append(np.array(self.pre_img_temp))
                self.batch_main_image.append(np.array(self.main_img_temp))
                self.batch_post_image.append(np.array(self.post_img_temp))
                self.batch_main_annot.append(np.array(self.main_annot_temp))
                # print(np.count_nonzero(self.main_annot_temp[:][:, :][:, :, 0]))

            else:
                self.pre_img_temp = np.array(
                    self.file_type_check_and_load_negative(pre_file_dict['pre_image']))
                self.main_img_temp = np.array(
                    self.file_type_check_and_load_negative(main_file_dict['main_image']))
                self.post_img_temp = np.array(
                    self.file_type_check_and_load_negative(post_file_dict['post_image']))
                self.main_annot_temp = np.zeros((512, 512, 1))

                self.pre_img_temp, self.main_img_temp, self.post_img_temp, self.main_annot_temp = image_crop.random_crop_center(
                    self.pre_img_temp, self.main_img_temp, self.post_img_temp, self.main_annot_temp, 200, 200)

                self.batch_pre_image.append(np.array(self.pre_img_temp))
                self.batch_main_image.append(np.array(self.main_img_temp))
                self.batch_post_image.append(np.array(self.post_img_temp))
                self.batch_main_annot.append(np.array(self.main_annot_temp))

        self.batch_pre_image = np.array(self.batch_pre_image)
        self.batch_main_image = np.array(self.batch_main_image)
        self.batch_post_image = np.array(self.batch_post_image)
        self.batch_main_annot = np.array(self.batch_main_annot)

        return self.batch_pre_image, self.batch_main_image, self.batch_post_image, self.batch_main_annot, self.batch_file_list
