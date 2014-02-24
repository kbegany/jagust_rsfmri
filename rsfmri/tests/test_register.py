
import os
from glob import glob
import tempfile
import unittest
import logging
import numpy as np
import numpy.testing as npt

import nibabel as ni

from .. import register

class RegistrationTest(unittest.TestCase):
    def setUp(self):
        ## artificial data
        tmpdir = tempfile.mkdtemp()
        files = []
        for item in (2,4):
            tmpdat = np.zeros((10,10,10))
            tmpdat[item:item+2, item:item+2,item:item+2] = item
            tmpimg = ni.Nifti1Image(tmpdat, np.eye(4))
            tmpf = os.path.join(tmpdir, 'file_%02d.nii.gz'%item)
            tmpimg.to_filename(tmpf)
            files.append(tmpf)
        self.files = files

    def tearDown(self):
        ## delete files
        pth, _ = os.path.split(self.files[0])
        for f in glob(os.path.join(pth, '*')):
            os.unlink(f)

    def test_function_logger(self):
        tmpdir, _ = os.path.split(self.files[0])
        register.function_logger(tmpdir)
        logger = logging.getLogger('rsfmri.register.test')
        message = 'testing function logger'
        logger.info(message)
        logf = glob(os.path.join(tmpdir, '*.log'))[0]
        npt.assert_equal(os.path.isfile(logf), True)
        loginfo = open(logf).read()
        npt.assert_equal(message in loginfo, True)

    def test_make_mean(self):
        newmean = register.make_mean(self.files)
        npt.assert_equal('mean_' in newmean, True)
        tmpdat = ni.load(newmean).get_data()
        npt.assert_equal(tmpdat.max(), 2)
        npt.assert_equal(tmpdat.min(), 0)

    def test_make_affine_filename(self):
        newname = register.make_aff_filename(self.files[0],
                self.files[1])
        npt.assert_equal(newname, 'file_04_to_file_02')
        newname = register.make_aff_filename('target.nii', 
                'moving.nii.gz')
        npt.assert_equal(newname, 'moving_to_target')

    def test_affine_register_cc(self):
        # test if you dont have paths to images
        npt.assert_raises(IOError, register.affine_register_cc,
                'file_a.nii', 'file_b.nii')
        # test non-matching base dirs
        npt.assert_raises(IOError, register.affine_register_cc,
                '/dira/sometarget.nii','/dirb/somemoving.nii')
        # test capture failure at ANTS level (eg nonexistant files)
        jnk = register.affine_register_cc('/notatarget.nii', 
                '/notamoving.nii')
        npt.assert_equal(jnk, None)
        tmpdir,_ = os.path.split(self.files[0])
        affine_file = os.path.join(tmpdir,  
                register.make_aff_filename(self.files[0],self.files[1])\
                        + 'Affine.txt')
        res = register.affine_register_cc(self.files[0], self.files[1])
        npt.assert_equal(res, affine_file)
        xfm_data = open(res).read().split('\n')
        npt.assert_equal(xfm_data[0], '#Insight Transform File V1.0')
        npt.assert_equal(xfm_data[-2], 'FixedParameters: -2.5 -2.5 2.5')
        npt.assert_equal('Parameters: 0.999' in xfm_data[3], True)
        
    def test_make_outfile(self):
        _, tmpf = os.path.split(self.files[0])
        outf = register.make_outfile(tmpf)
        npt.assert_equal(outf, 'xfm_' + tmpf)
        outf = register.make_outfile(tmpf, inverse = True)
        npt.assert_equal(outf, 'invxfm_' + tmpf)

    def test_reslice(self):
        moving, targetspace = self.files[:2]
        resliced = register.reslice(moving, targetspace )
        npt.assert_equal('rsl_' in resliced, True)
        npt.assert_equal(ni.load(resliced).get_affine(), 
            ni.load(targetspace).get_affine())
        # make other space
        tmpdir, _ = os.path.split(moving)
        tmpdat = np.zeros((15,15,15))
        tmpimg = ni.Nifti1Image(tmpdat, np.eye(4))
        tmpf = os.path.join(tmpdir, 'reslice_test.nii.gz')
        tmpimg.to_filename(tmpf)
        resliced = register.reslice(tmpf, targetspace)
        npt.assert_equal(ni.load(resliced).get_affine(), 
            ni.load(targetspace).get_affine())

    def test_apply_transform(self):
        moving, target = self.files
        tmpdir,_ = os.path.split(moving)
        affine_file = os.path.join(tmpdir,  
                register.make_aff_filename(moving, target)\
                        + 'Affine.txt')
        ## get transform
        xfm = register.affine_register_cc(target, moving)
        #apply_transform(moving, transform, outfile=None, target=None, use_nn = False)
        xfm_filenn = register.apply_transform(moving, xfm, use_nn=True)
        npt.assert_array_equal(ni.load(target).get_data()>0, 
            ni.load(xfm_filenn).get_data()>0)
        xfm_file = register.apply_transform(moving, xfm)
        ## divide by 2 and round both to handle interpolation
        npt.assert_array_almost_equal(np.round(ni.load(target).get_data()/ 2.), 
            np.round(ni.load(xfm_file).get_data()), decimal=2)
        ## test inverse
        invxfm_nn = register.apply_transform(target, '-i %s'%xfm, use_nn=True)
        npt.assert_equal('invxfm' in invxfm_nn, True)
        npt.assert_equal(ni.load(moving).get_data()>0,
            ni.load(invxfm_nn).get_data()>0)

    def test_grow_mask(self):
        infile, othermask = self.files
        # test odd size for kernel
        with npt.assert_raises(ValueError):
            register.grow_mask(infile, size = 2)
        grown = register.grow_mask(infile)
        npt.assert_equal(ni.load(infile).get_data().sum() < \
            ni.load(grown).get_data().sum(), True )
        grown5 = register.grow_mask(infile, size=5)
        npt.assert_equal(ni.load(grown5).get_data().sum() > \
            ni.load(grown).get_data().sum(), True )
        grown = register.grow_mask(infile, othermask, 5)
        ## mask is larger, but mask region nulls it out
        npt.assert_equal(ni.load(infile).get_data().sum() > \
            ni.load(grown).get_data().sum(), True )        

