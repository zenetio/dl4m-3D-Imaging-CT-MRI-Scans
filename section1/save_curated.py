"""
This file contains code that will kick off training and testing processes
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from shutil import copyfile

def two_largest(inlist):
    """Return the two largest items in the sequence. The sequence must
    contain at least two items."""
    largest = second_largest = 0
    it1 = it2 = 0

    for i,item in enumerate(inlist):
        if item > largest:
            largest = item
            it1 = i
        elif largest > item > second_largest:
            second_largest = item
            it2 = i
    # Return the results as a tuple
    return largest, it1, second_largest, it2

def copyFiles(img, lbl):
    """" Save all n curated images and labels to out directory
         Create directories is does not exist yet
         Args:
         img: list of image files
         lbl: list of label files
    """
    if not os.path.exists(os.path.join(out_root_dir, "images")):
        os.makedirs(os.path.join(out_root_dir, "images"))
    if not os.path.exists(os.path.join(out_root_dir,"labels")):
        os.makedirs(os.path.join(out_root_dir, "labels"))
    # copy
    for i,f in enumerate(img):
        img_dstdir = os.path.join(out_root_dir, "images", os.path.basename(f))
        lbl_srcdir = os.path.join(in_root_dir, "labels", os.path.basename(f))
        lbl_dstdir = os.path.join(out_root_dir, "labels", os.path.basename(f))
        
        # copy images and labels
        try:
            # copy image
            copyfile(f, img_dstdir)
        except:
            print(f"Error trying copy image file {f}")
        
        try:
            # copy label
            copyfile(lbl_srcdir, lbl_dstdir)
        except:
            print(f"Error trying copy label file {lbl_srcdir}")


in_root_dir = r"./data/TrainingSet"
out_root_dir = "./section1/out/TrainingSet"
# where am I?
# x = os.getcwd()

# Load an image and a segmentation mask into variables called image and label
path = r"/data/TrainingSet"
dirs = np.array([[(os.path.join(dp, f), nib.load(os.path.join(dp, f))) for f in files]
                   for dp,_,files in os.walk(in_root_dir) if len(files) != 0])

# check if all image files has a label in label directory 
# rebuild list of files
new_images = []
new_labels = []

if len(dirs[0]) != len(dirs[1]):
    # we have a problem
    # create a new list
    new_list = None
    if len(dirs[0]) > len(dirs[1]):
        # there is a missing label file
        new_list = dirs[1]
    else:
        # there is a missing image file
        new_list = dirs[0]

    for f in new_list:
        img_dir = os.path.join(in_root_dir, "images", os.path.basename(f[0]))
        lbl_dir = os.path.join(in_root_dir, "labels", os.path.basename(f[0]))
        if os.path.exists(img_dir) and os.path.exists(lbl_dir) and \
            img_dir != '.' and lbl_dir != '.':
            new_images.append(img_dir)
            new_labels.append(lbl_dir)
else:
    new_images = dirs[0]
    new_labels = dirs[1]

# We have the same number of files for images and labels
assert(len(new_images) == len(new_labels))

# Now check for outliers
images_names = [nib.load(f) for f in new_images]
images = [np.stack(s.get_fdata()) for s in images_names]
labels_names = [nib.load(f) for f in new_labels]
labels = [np.stack(s.get_fdata()) for s in labels_names]

vals, bins, ignored = plt.hist(images, bins = 1)
# based on the histogram we can see two outliers. Lets find them and plot a new histogram
v1,i1,v2,i2 = two_largest(vals)
# remove outliers from lists
img_curated = []
lbl_curated = []
for i in range(len(vals)):
    if i != i1 and i != i2:
        img_curated.append(new_images[i])
        lbl_curated.append(new_labels[i])

# plot the new histogram
vals2, bins, ignored = plt.hist(img_curated, bins = 1)
# Yeah, now the histogram is much better!

# Save files to out directory
copyFiles(img_curated, lbl_curated)

    

