import os
import numpy as np
import nibabel as nib


def load_BRATS_data(path):
    t1 = []
    t1gd = []
    t2 = []
    flair = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if(file.endswith('t1.nii.gz')):
                t1.append(nib.load(filepath).get_fdata())
            elif(file.endswith('t1Gd.nii.gz')):
                t1gd.append(nib.load(filepath).get_fdata())
            elif(file.endswith('t2.nii.gz')):
                t2.append(nib.load(filepath).get_fdata())
            elif(file.endswith('flair.nii.gz')):
                flair.append(nib.load(filepath).get_fdata())
            elif(file.endswith('GlistrBoost_ManuallyCorrected.nii.gz')):
                labels.append(nib.load(filepath).get_fdata())

    return t1, t1gd, t2, flair, labels


def mode(array):
    counts = dict()
    array_flat = array.flatten()
    
    for a in array_flat:
        if a == 0:
            continue
        if a not in counts:
            counts[a] = 1
        else:
            counts[a] += 1
            
    maxCount = 0
    maxElement = None
    for c in counts:
        if(counts[c] >= maxCount):
            maxElement = c
            maxCount = counts[c]
    return maxElement


"""
in-place data processing
It subtracts the mode of the data, and normalize standard deviation to 1
"""
def preprocess(data):
	epsilon = 1e-6
	data -= mode(data)
	data = data / (data.std + epsilon)
