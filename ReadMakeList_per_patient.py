"""
#---------------------------*[Description]*---------------------------------#
#           * This is the main code for BM 3ch to 3 vgg model
#===========================================================================#
#------------------------*[Record of version]*------------------------------#
#---------------------------------------------------------------------------#
#     Date         |   Name     |           Version description             #
#---------------------------------------------------------------------------#
# 2019.01.28                         cross validation call data per case
#---------------------------------------------------------------------------#
"""
import os, glob
import numpy as np
import nibabel
import tensorflow as tf
import pickle

def pause():
    input('Press the <Enter> key to continue...')


def make_or_patient_list_of_pickle(Par_data_dir):
    pickle_name = '3cto3m_cross_validation_patient_list.pickle'

    pickle_file_path = os.path.join(Par_data_dir, pickle_name)

    if not os.path.exists(pickle_file_path):
        print(print('There is no pickle file. \n'))
        print('We are in the data listing step. \n')

        list_of_data = create_patient_list(os.path.join(Par_data_dir))

        print("Pickling ...")
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(list_of_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Pickle file already exists! I will use this .pickle file \n')

    with open(pickle_file_path, 'rb') as f:
        result = pickle.load(f)
        pickle_records = result['final_BM_data']

        del result

    return pickle_records



def create_patient_list(Par_dir):
    if not tf.gfile.Exists(Par_dir):
        print("Could not find the 'input' directory. Set again. \n")
        return None

    sub_par_name_in_Par_dir = ['final_BM_data']

    sub_par_directory_path_list = []
    img_annot_list = {}

    for sub_par_name in sub_par_name_in_Par_dir:
        Par_dir_list = []
        FilePath_list = []
        FileName_list = []

        for ParName, SubName, FileName in os.walk(os.path.join(Par_dir, sub_par_name)):
            if FileName != []:
                Par_dir_list.append(ParName)

        # print('Par_dir_list:',Par_dir_list)
        # print(len(Par_dir_list))

        return Par_dir_list








