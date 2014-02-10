import os
from glob import glob
import tempfile
import unittest
import logging
import numpy as np
import numpy.testing as npt

import nibabel as ni

from .. import transform


def get_data():
    pth, _ = os.path.split(__file__)
    testdir = os.path.join(pth, 'data')
    globstr = os.path.join(testdir, 'B*Affine.txt')
    affines = sorted(glob(globstr))
    return affines


def test_collate_affines():
    filelist = get_data()
    move_params = transform.collate_affines(filelist)
    npt.assert_equal(len(filelist), move_params.shape[0])
    expected = np.array([
        -0.0165277 , 
        -0.00147508, 
        -0.0356341 ,  
        0.0012551 ,  
        0.00040574,
        -0.00011537,  
        0.10242811
        ])
    npt.assert_almost_equal(move_params[0],expected)
