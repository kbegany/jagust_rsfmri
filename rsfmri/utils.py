import os
import datetime
from glob import glob

import numpy as np

import nibabel

########################

def get_files(dir, globstr):
    """
    uses glob to find dir/globstr
    returns sorted list; number of files
    """
    searchstr = os.path.join(dir, globstr)
    files = glob(searchstr)
    files.sort()
    return files, len(files)


def make_datestr():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%S')


def get_slicetime(nslices):
    """
    If TOTAL # SLICES = EVEN, then the excitation order when interleaved
    is EVENS first, ODDS second.
    If TOTAL # SLICES = ODD, then the excitation order when interleaved is
    ODDS first, EVENS second.
    """
    if np.mod(nslices,2) == 0:
        sliceorder = np.concatenate((np.arange(2,nslices+1,2),
                                     np.arange(1,nslices+1,2)))
    else:
        sliceorder = np.concatenate((np.arange(1,nslices+1,2),
                                     np.arange(2,nslices+1,2)))
    return sliceorder
        

def get_slicetime_vars(infiles, TR=None):
    """
    uses nibabel to get slicetime variables
    """
    if hasattr('__iter__', infiles):
        img = nibabel.load(infiles[0])
    else:
        img = nibabel.load(infiles)
    hdr = img.get_header()
    if TR is None:
        raise RuntimeError('TR is not defined ')
    
    shape = img.get_shape()
    nslices = shape[2]
    TA = TR - TR/nslices
    sliceorder = get_slicetime(nslices)
    return dict(nslices=nslices,
                TA = TA,
                TR = TR,
                sliceorder = sliceorder)


