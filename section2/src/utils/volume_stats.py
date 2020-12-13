"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np
# from sklearn.metrics import jaccard_score

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    af = a.flatten()
    bf = b.flatten()
    intersection = np.sum((af>0) * (bf>0))
    volumes = np.sum(af>0) + np.sum(bf>0)
    if volumes == 0:
        return -1
    return 2.*float(intersection) / float(volumes)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # jaccard = intersection / union
    af = a.flatten()
    bf = b.flatten()
    af=a
    bf=b
    intersection = np.sum((af>0) * (bf>0))
    union = (np.sum(af>0) + np.sum(bf>0)) - intersection
    return intersection / union