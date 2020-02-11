# CT_data_segmentation

Detecting Bone metastasis in vertebrae using convolution neural network.

## Requirements

- Python 3.5
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [nibabel](https://github.com/nipy/nibabel)
- [pydicom](https://github.com/pydicom/pydicom)
- [imageio](https://github.com/imageio/imageios)
- [scipy](https://github.com/scipy/scipy)
- [TensorFlow 1.3.0](https://github.com/tensorflow/tensorflow)

## Usage

First download Backbone[imagenet-vgg-verydeep-19.mat] with:

    $ http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

Dataset structure (5fold cross validation):

    data
    ├── Fold1 (100 Patients/Fold)
    |   ├── 102024_aaaaa (patient ID)
    |   |    ├── 00000000.nii (Mask image, Ground Truth)
    |   |    ├── ser000img00000.dcm (Dicom image)
    |   |    └── ...
    |   ├── 102024_bbbbb        
    |   |    ├── 11111111.nii 
    |   |    ├── ser111img11111.dcm 
    |   |    └── ...
    |   ├── 102024_ccccc 
    |   └── ...
    |
    ├── Fold2 
    ├── ...
    ├── Fold5
    └── Osteolytic_pickle

The image patch is used for the training, extracting x13 patches from one slice.
The structure used for the train is a modified vgg-19 model, adding three deconvolutional layers more to obtain the same size of the output image.

From the idea that radiologists read CT data by scrolling and comparing consecutive CT slices, we designed the model using three continuous slice feeding each model.


To train a model(use your 'data_Par_dir'):

    $ python Main_code.py 

Utilize the Osteolytic_pickle for the model fine-tuning.
    
## Results

### Output(512x512) of `learning rate=1×10^(-4)` 10 epochs and transfer learning `learning rate=1×10^(-5)` for 5 epochs additional.
![Prediction_image_Example](https://user-images.githubusercontent.com/55068090/69806563-86eda580-1226-11ea-8365-472374396db9.png)
