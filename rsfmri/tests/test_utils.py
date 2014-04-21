import os
import re
import unittest
import tempfile

import numpy as np
import numpy.testing as npt

import nibabel
from  .. import utils


# create some test data
def make_test_data(fill = 0):
    tmpdir = tempfile.mkdtemp()
    tmpnii = os.path.join(tmpdir, 'tmpfile.nii.gz')
    nslices = 6
    dat = np.zeros((10,10,nslices)) + fill
    img = nibabel.Nifti1Image(dat, np.eye(4))
    img.to_filename(tmpnii)
    return nslices, tmpnii

def test_get_files():
    pth, _ = os.path.split(__file__)
    myinit = os.path.join(pth, '__init__.py')
    res, nfiles = utils.get_files(pth, '*')
    npt.assert_equal(myinit in res, True)


def test_make_dir():
    tmpdir = tempfile.mkdtemp()
    dirnme = 'Created_directory'
    newdir, exists = utils.make_dir(tmpdir, dirnme)
    npt.assert_equal(newdir, os.path.join(tmpdir, dirnme))
    npt.assert_equal(exists, False)
    newdir, exists = utils.make_dir(tmpdir, dirnme)
    npt.assert_equal(exists, True)


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
    # cleanup
    os.unlink(tmpnii)
    os.unlink(tmpfile)

def test_realign_unwarp():
    ru = utils.nipype_ext.RealignUnwarp()
    npt.assert_raises(ValueError, ru.run)

def test_make_mean():
    _, one_file = make_test_data(fill = 1)
    _, two_file = make_test_data(fill = 2)
    pth = os.path.dirname(one_file)
    outfile = os.path.join(pth, 'meanfile.nii.gz')
    mean_file = utils.make_mean([one_file, two_file], outfile)
    npt.assert_equal(outfile, mean_file)
    dat = nibabel.load(mean_file).get_data()
    npt.assert_equal(dat.mean(), 1.5)
    npt.assert_raises(IOError, utils.make_mean, 'stupidfile.nii',
                      outfile)
    os.unlink(one_file)
    os.unlink(two_file)

def test_aparc_mask():
    # make label dat
    test_dat = np.zeros((10,10,10))
    test_dat[:3,:3, :] = 3
    test_dat[3:8,3:8,3:8] = 8
    labels = (3,8)
    # make nii
    tmpdir = tempfile.mkdtemp()
    tmpnii = os.path.join(tmpdir, 'tmpfile.nii.gz')
    img = nibabel.Nifti1Image(test_dat, np.eye(4))
    img.to_filename(tmpnii)
    # test
    tmpmask = os.path.join(tmpdir, 'tmpmask.nii.gz')
    tmpmask = utils.aparc_mask(aparc=tmpnii, labels=labels, outfile=tmpmask)
    mask_data = nibabel.load(tmpmask).get_data() # get data from new mask
    npt.assert_equal(mask_data>0, test_dat>0)
    # cleanup
    os.unlink(tmpnii)
    os.unlink(tmpmask)

def test_extract_seed_ts():
    # make nii
    myrand = np.random.RandomState(42)
    tmpdir = tempfile.mkdtemp()
    tmp4d = os.path.join(tmpdir, 'tmpnii4d.nii.gz')
    dat = myrand.random_sample((10,10,10,40))
    img = nibabel.Nifti1Image(dat, np.eye(4))
    img.to_filename(tmp4d)
    # make seeds
    tmpseed = os.path.join(tmpdir, 'tmpseed.nii.gz')
    seed_dat = np.zeros((10,10,10))
    seed_dat[:5,:5,:] = 1
    seedimg = nibabel.Nifti1Image(seed_dat, np.eye(4))
    seedimg.to_filename(tmpseed)
    # test mean
    meants = utils.extract_seed_ts(tmp4d, [tmpseed])
    npt.assert_equal(isinstance(meants, list), True)
    npt.assert_almost_equal(meants[0][0], 0.49458008)
    npt.assert_equal(meants[0].shape, (40,))
    # test missing
    dat[0,0,:] = 0
    img = nibabel.Nifti1Image(dat, np.eye(4))
    #img.to_filename(tmpnii)
    # cleanup
    #os.unlink(tmpnii)
    os.unlink(tmpseed)
    os.unlink(tmp4d)

def test_zero_pad_movement():
    # create generic movement file
    myrand = np.random.RandomState(42)
    origdat = myrand.random_sample((10,7))
    colnames = ['col{}'.format(x) for x in range(7)]
    df = utils.pandas.DataFrame(origdat, columns = colnames)
    paddat = utils.zero_pad_movement(df)
    npt.assert_equal(df.values, paddat.values[1:,:])