"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape

def LoadHippocampusData(root_dir, x_shape, y_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        try:
            image, mh = load(os.path.join(image_dir, f))
        except:
            print(f"Error when loading: {image_dir + '/' + f}")
            continue        # cannot continue processing this file
        try:
            label, lh = load(os.path.join(label_dir, f))
        except:
            print(f"Error when loading: {label_dir + '/' + f}")
            continue        # cannot continue processing this file

        # normalize all images (but not labels) so that values are in [0..1] range
        image = (image.astype(np.single) - np.min(image))/(np.max(image) - np.min(image))
        #image = image.astype(np.single)/np.max(image)

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here
        
        # assume image slice size
        image_slice = image.shape[2]
        # check if slices are different and assume minimum value
        if image.shape[2] != label.shape[2]:
            image_slice = min(image.shape[2], label.shape[2])
            print(f"{f} shape: {image.shape} != label shape: {label.shape}")
        #print(f"{f}, shape: {image.shape}, label shape: {label.shape}")    
        new_image = med_reshape(image, new_shape=(x_shape, y_shape, image_slice))
        new_label = med_reshape(label, new_shape=(x_shape, y_shape, image_slice)).astype(int)

        # Why do we need to cast label to int?
        # ANSWER: loss function only accepts int values

        out.append({"image": new_image, "seg": new_label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[2] for x in out])} slices")
    return np.array(out)


def LoadHippocampusData2(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    #i=0
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        try:
            image, mh = load(os.path.join(image_dir, f))
        except:
            print(f"Error when loading: {image_dir + '/' + f}")
            continue        # cannot continue processing this file
        try:
            label, lh = load(os.path.join(label_dir, f))
        except:
            print(f"Error when loading: {label_dir + '/' + f}")
            continue        # cannot continue processing this file

        # normalize all images (but not labels) so that values are in [0..1] range
        image = (image.astype(np.single) - np.min(image))/(np.max(image) - np.min(image))

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here
        
        new_image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        new_label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # Why do we need to cast label to int?
        # ANSWER: loss function only accepts int values

        out.append({"image": new_image, "seg": new_label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
