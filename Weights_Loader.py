"""
#------------------------*[Record of version]*---------------------------#
#           Model weight Loader (imagenet-vgg-verydeep-19.mat)
#------------------------------------------------------------------------#
"""
import TensorFlowUtils_offline as utils
import numpy as np

def pause():
    input('Press the <Enter> key to continue...')


def weights_loader(model_dir, model_file_name):
    model_data = utils.get_model_data(model_dir, model_file_name)
    weights = np.squeeze(model_data['layers'])
    normal_values = model_data['meta']['normalization'][0][0]['averageImage'][0][0]

    return weights, normal_values
