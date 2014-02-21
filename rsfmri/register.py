
import os
import sys

from glob import glob
import logging
import datetime

import numpy as np
from scipy.signal import convolve
import nibabel as ni
from nipype.interfaces.base import CommandLine
from nipype.utils.filemanip import (split_filename, fname_presuffix)


ANTSPATH='/home/jagust/fmri-pstask/pilot/ANTS/ANTs-1.9.v4-Linux/bin'

def timestr():
    return datetime.datetime.strftime(datetime.datetime.now(), 
            '%Y-%m-%d-%H-%M')

def function_logger(indir, console = False):
    logfile = os.path.join(indir, 'logger_%s.log'%(timestr()))
    logging.basicConfig(filename=logfile,level=logging.DEBUG)
    

    logger = logging.getLogger('rsfmri.register')
    logger.setLevel(logging.DEBUG)

    fileh = logging.FileHandler(logfile)
    fileh.setLevel(logging.DEBUG)
    consoleh = logging.StreamHandler()
    consoleh.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)
    consoleh.setFormatter(formatter)

    logger.addHandler(fileh)
    if console:
        logger.addHandler(consoleh)


def threshold_subslice(img_dat):
    """ after the small functional scan is mapped to the larger
    image space, the zero values are filled with non-zero values
    so these need to be set back to zero for masking to work"""
    height, value = np.histogram(img_dat.flatten(),100)
    thr = value[height.argmax()]
    img_dat[img_dat < thr+10] = 0
    return img_dat

def grow_mask(infile, othermask=None, size=3):
    """given a 3d volume infile, grow the region using a
    3X3X3 kernel, optionally mask with othermask (0 == 0)

    Parameters
    ==========
    infile : string
        filename of 3d volume
    othermask : string or None
        filename of optianl image used as mask
    size : int
        size of symmetric kernel, must be odd number 
    """
    img = ni.load(infile)
    dat = img.get_data().squeeze()
    #dat = threshold_subslice(dat)
    if not size%2 ==1:
        raise ValueError('Size must be odd integer, not %d'%size)
    kernel = np.ones((size,) * len(dat.shape))
    result = convolve(dat, kernel, 'same')
    newdat = np.array(result>0, dtype = int)
    if othermask is not None:
        msk = ni.load(othermask)
        if not msk.get_shape() == img.get_shape():
            raise IOError('shape mismatch, img: %s  mask: %s'%(img.get_shape(),
                    msk.get_shape()))
        newdat[msk.get_data() == 0] = 0
    outf = fname_presuffix(infile, prefix='mask_k%d'%size)
    newimg = ni.Nifti1Image(newdat, img.get_affine())
    newimg.to_filename(outf)
    return outf


def make_mean(niftilist, prefix='mean_'):
    """given a list of nifti files
    generates a mean image"""
    logger = logging.getLogger('antsregister')
    n_images = len(niftilist)
    newfile = fname_presuffix(niftilist[0], prefix=prefix)
    affine = ni.load(niftilist[0]).get_affine()
    shape =  ni.load(niftilist[0]).get_shape()
    newdat = np.zeros(shape)
    for item in niftilist:
        newdat += ni.load(item).get_data().copy()
    newdat = newdat / n_images
    newdat = np.nan_to_num(newdat)
    newimg = ni.Nifti1Image(newdat, affine)
    newimg.to_filename(newfile)
    logger.info(niftilist)
    logger.info(newfile)
    return newfile



def make_aff_filename(target, moving):
    pth, tnme, ext = split_filename(target)
    pth, mnme, ext = split_filename(moving)
    ## no dir defined
    outf = '%s_to_%s'%(mnme, tnme)
    return outf

def affine_register_cc(target, moving):
    """ uses ants to create an affine mapping using cross correlation
    with only rigid transforms"""
    logger = logging.getLogger('antsregister.affine_register_cc')
    scriptdir = os.getcwd()
    tpth, tnme = os.path.split(target)
    mpth, mnme = os.path.split(moving)
    if not tpth == mpth:
        logger.error('target and moving in different dirs: %s %s'%(target, moving))
        raise IOError('target and moving need to be in same dir'\
                't: %s, m: %s'%(target, moving))
    if tpth == '':
        raise IOError('need to specify location of files, not %s'%tpth)

    os.chdir(tpth)
    _basecmd = os.path.join(ANTSPATH,'ANTS')
    dim = '3'
    # make command
    # CC [target,moving,1,5] # no spaces!
    # cross-correl target moving weight region_radius
    similarity = ''.join(['CC', '[', tnme, ',', mnme, ',', 
        '1',',','5', ']'])
    outfile_prefix = make_aff_filename(target, moving)
    fullcmd = ' '.join([_basecmd, dim, '-m', similarity, '-i', '0', 
        '-o', outfile_prefix])
    logging.info(fullcmd)
    res = CommandLine(fullcmd, ignore_exception=True).run()
    os.chdir(scriptdir)

    if  res.runtime.returncode == 0 and not 'Exception' in res.runtime.stderr:
        #logger.info(res.runtime.stdout)
        logger.info(fullcmd)
        logger.info(os.path.join(tpth, outfile_prefix + 'Affine.txt'))
        return os.path.join(tpth, outfile_prefix + 'Affine.txt')
    else:
        logger.error(res.runtime.stderr)
        return None


def affine_register_mi(target, moving, mask=None):
    """ uses ANTS to create an affine mapping between different 
    modalities of the same subjects anatomy, 
    uses mututal information """
    logger = logging.getLogger('antsregister.affine_register_mi')
    scriptdir = os.getcwd()
    tpth, tnme = os.path.split(target)
    mpth, mnme = os.path.split(moving)
    if not tpth == mpth:
        raise IOError('target and moving need to be in same dir'\
                't: %s, m: %s'%(target, moving))
    if tpth == '':
        raise IOError('need to specify location of files, not %s'%tpth)

    os.chdir(tpth)
    _basecmd = os.path.join(ANTSPATH,'ANTS')
    dim = '3'
    # specify command line options
    # MI [target,moving,1,32] #no witespaces!
    # mutualinfo  target moving weight  number_hist_bins
    similarity = ''.join(['MI', '[', tnme, ',', mnme, ',', 
        '1',',','32', ']'])
    outfile_prefix = make_aff_filename(target, moving)
    fullcmd = ' '.join([_basecmd, dim, '-m', similarity, '-i', '0', 
        '-o', outfile_prefix])
    if mask is not None:
        maskpth, masknme = os.path.split(mask)
        fullcmd = ' '.join([fullcmd, '-x', masknme])
    logger.info(fullcmd)
    res = CommandLine(fullcmd).run()
    os.chdir(scriptdir)

    if  res.runtime.returncode == 0 and not 'Exception' in res.runtime.stderr:
        logger.info(fullcmd)
        logger.info(os.path.join(tpth, outfile_prefix + 'Affine.txt'))
        return os.path.join(tpth, outfile_prefix + 'Affine.txt')
    else:
        logger.error(fullcmd)
        logger.error(res.runtime.stderr)
        return None


def n4_biascorrect(filename):
    """N4BiasFieldCorrection $DIM -i $MOVING -o ${OUTPUTNAME}repaired.nii.gz 
    -s 2 -c [50x50x50x50,0.000001] -b [200]
    """
    _basecmd = os.path.join(ANTSPATH, 'N4BiasFieldCorrection')
    logger = logging.getLogger('antsregister.n4_biascorrect')
    scriptdir = os.getcwd()
    
    outfile = fname_presuffix(filename, 'n4_')
    tpth, tnme = os.path.split(filename)
    _, outnme = os.path.split(outfile)
    os.chdir(tpth)
    cmd = ' '.join([
        _basecmd, 
        '3',
        '-i',
        tnme,
        '-o',
        outnme,
        '-s',
        '2',
        '-c',
        '[50x50x50x50,0.000001]',
        '-b',
        '[200]'
        ])
    logger.info(cmd)
    res = CommandLine(cmd).run()
    os.chdir(scriptdir)

    if  res.runtime.returncode == 0 and not 'Exception' in res.runtime.stderr:
        
        logger.info(outfile)
        return outfile
    else:
        
        logger.error(res.runtime.stderr)
        return None
    pass


def make_outfile(filename, prefix = 'xfm_', inverse = False):
    if inverse:
        prefix = 'inv' + prefix
    return fname_presuffix(filename, prefix)


def reslice(moving, targetspace, nearestn=False):
    """ reslice data in moving to space of targetspace image
    default linear interpolation unless nearestn is True 
    then uses nearest neighbor"""
    _basecmd = os.path.join(ANTSPATH, 'WarpImageMultiTransform')
    logger = logging.getLogger('antsregister.reslice')
    nn = '--use-BSpline'
    if nearestn:
        nn = '--use-NN'
    outfile = make_outfile(moving, prefix = 'rsl_' + nn.replace('--use-','') )
    cmd =  ' '.join([ _basecmd, 
                      '3',
                      moving,
                      outfile, 
                      '-R',
                      targetspace,
                      '--reslice-by-header',
                      nn])
    res = CommandLine(cmd).run()
    if res.runtime.returncode == 0:
        logger.info(cmd)
        return outfile
    else:
        logger.error(cmd)
        logger.error(res.runtime.stderr)
        return None


def apply_transform(moving, transform, outfile=None, target=None, use_nn = False):
    """apply a transform or series of transforms to moving

    Parameters
    ==========
    moving : filename
        filename of images transformes are being applied to
    transform : string
        string indicating transforms 
        examples
        'a_to_b.Affine.txt'  simple affine
        ' -i a_to_b.Affine.txt' inverse affine
        'a2b_Warp.nii.gz a2b_Affine.txt' warp image to template
        '-i a2b_Affine.txt a2b_InverseWarp.nii.gz' warp template to image
    outfile : filename
        specify filename of output, if None it will be generated
    target : filename
        filename of image to use for resliced space
    use_nn : bool
        if True use nearest neighbor interpolation, default is False
        using linear interpolation

    Returns
    =======
    outfile : filename
        name of transformed file, None if fails
    """
    _basecmd = os.path.join(ANTSPATH, 'WarpImageMultiTransform')
    logger = logging.getLogger('antsregister.apply_transform')
    scriptdir = os.getcwd()
    pth, _ = os.path.split(moving)
    os.chdir(pth)
    dim = '3'
    ## interpolation 3rd degree bspline is default
    nn = '--use-NN' if use_nn else '--use-BSpline'
    if outfile is None:
        prefix = 'invxfm_' if '-i' in transform else 'xfm_'
        outfile = fname_presuffix(moving, prefix)
    if target is not None:
        fullcmd = ' '.join([_basecmd, dim, moving, outfile, 
            '-R', target, transform, nn])
    else:
        fullcmd = ' '.join([_basecmd, dim, moving, outfile, 
            transform, nn])
    logging.info(fullcmd)
    res = CommandLine(fullcmd).run()
    os.chdir(scriptdir)
    if  res.runtime.returncode == 0:
        logger.info(fullcmd)
        return outfile 
    else:
        logger.error(res.runtime.stderr)
        return None

   



if __name__ =='__main__':

    datadir = '/home/jagust/fmri-pstask/pilot/B13-003/test_coreg/func'
    globstr = os.path.join(datadir, 'B13-003_multib_PA*.nii.gz')
    allf = sorted(glob(globstr))
    target = allf[0]
    basedir, _ = os.path.split(target)
    function_logger(basedir)
    for movin in allf[1:]:
        tmpaff = affine_register_CC(target, movin)
        if tmpaff is None:
            continue
        moved = apply_affine(movin, tmpaff)




