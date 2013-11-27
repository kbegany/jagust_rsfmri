import os
import re
import unittest
import tempfile

import numpy as np
import numpy.testing as npt

import nibabel
from  .. import utils


# create some test data
def make_test_data():
    tmpdir = tempfile.mkdtemp()
    tmpnii = os.path.join(tmpdir, 'tmpfile.nii.gz')
    nslices = 6
    img = nibabel.Nifti1Image(np.empty((10,10,nslices)),np.eye(4))
    img.to_filename(tmpnii)
    return nslices, tmpnii

def test_get_files():
    cwd = os.getcwd()
    myinit = os.path.join(cwd, '__init__.py')
    res, nfiles = utils.get_files(cwd, '*')
    npt.assert_equal(myinit in res, True)

def test_make_datestr():
    new_str = utils.make_datestr()
    parts = new_str.split('_')
    npt.assert_equal(len(parts), 5)
    g = re.search('[0-9]{4}', parts[0])
    npt.assert_equal(g  == None, False)

def test_get_slicetime():
    # test even
    slicetime = utils.get_slicetime(6)
    npt.assert_equal(slicetime, [2,4,6,1,3,5])
    # test odd
    slicetime = utils.get_slicetime(5)
    npt.assert_equal(slicetime, [1,3,5,2,4])
    ## assert slicetime is a list
    npt.assert_equal(type(slicetime), type([]))

def test_get_slicetime_vars():
    nslices, tmpnii = make_test_data()
    npt.assert_raises(RuntimeError, utils.get_slicetime_vars, tmpnii)
    TR = 2.2
    stdict = utils.get_slicetime_vars(tmpnii, TR)
    npt.assert_equal(stdict['sliceorder'], np.array([2, 4, 6, 1, 3, 5]))
    npt.assert_equal(stdict['TR'], TR)
    npt.assert_equal(stdict['nslices'], nslices)
    ## cleanup
    os.unlink(tmpnii)

def test_realign_unwarp():
    ru = utils.nipype_ext.RealignUnwarp()
    npt.assert_raises(ValueError, ru.run)

def test_save_json():
    nslices, tmpnii = make_test_data()
    stdict = utils.get_slicetime_vars(tmpnii, 2.2)
    tmpdir, _ = os.path.split(tmpnii)
    tmpfile = os.path.join(tmpdir, 'test_json.json')
    utils.save_json(stdict, tmpfile)
    npt.assert_equal(os.path.isfile(tmpfile), True)
    npt.assert_raises(IOError, utils.save_json, stdict, None)
    npt.assert_raises(IOError, utils.save_json, set([0,1,2]), tmpfile)
    # cleanup
    os.unlink(tmpnii)
    os.unlink(tmpfile)
    
def test_load_json():
    # make temp files
    nslices, tmpnii = make_test_data()
    TR = 2.2
    stdict = utils.get_slicetime_vars(tmpnii, TR)
    tmpdir, _ = os.path.split(tmpnii)
    tmpfile = os.path.join(tmpdir, 'test_json.json')
    utils.save_json(stdict, tmpfile)
    # test roundtrip
    tmpdict = utils.load_json(tmpfile)
    npt.assert_equal(tmpdict == stdict, True)
    npt.assert_equal(tmpdict['TR'], TR)
    # not a valid file
    npt.assert_raises(IOError, utils.load_json, 'notafile.txt')
    npt.assert_raises(IOError, utils.load_json, open(tmpfile))
