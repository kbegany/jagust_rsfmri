import os
import sys
from glob import glob
import argparse

import numpy as np

from rsfmri import utils
from rsfmri import register as reg
from rsfmri import transform

import logging


def split(rawdir, destdir, sid, logger, globstr='B*func4d.nii*'):
    """ split files into 3D vols,
    return is None if it fails"""
    files, nfiles = utils.get_files(rawdir, globstr)
    logger.info('SUBID: {0}, nfiles: {1}'.format(sid, nfiles))
    if not nfiles == 1:
        logger.error('Raw functional not found: {0}'.format(globstr))
        return None
    rawfunc = files[0]
    funcs =  utils.fsl_split4d(rawfunc, destdir, sid)
    return funcs


def make_realignst(funcs, TR, logger):
    """use spm to reaign and slicetime correct files"""
    stvars = utils.get_slicetime_vars(funcs, TR=TR)
    meanfunc, realigned, params = utils.spm_realign(funcs)
    if meanfunc is None:
        logger.error('{0}: spm realignment failed'%(funcs))
        return None, None
    st_realigned = utils.spm_slicetime(realigned, stdict = stvars)
    if st_realigned is None:
        logger.error('{0}: slice timing failed'.format(realigned))
        return None, None
    ## zip files
    utils.zip_files(funcs)
    utils.zip_files(realigned[1:]) # first is original func
    utils.zip_files(st_realigned)
    movement_array = np.loadtxt(params)
    return st_realigned, movement_array

def plot_write_movement(destdir, sid, movement, logger):
    transform.plot_movement(movement, destdir)
    outfile = os.path.join(destdir, '{0}_movement.xls'.format(sid))
    transform.movementarr_to_pandas(movement, outfile)



def process_subject(subdir, tr, logger, despike=False):
    """ process one subject despike (optional), realign and
    do slicetime correction via SPM tools"""
    globstr = 'B*func4d.nii*'
    workdirnme = 'spm_realign_slicetime'
    if despike:
        workdirnme = 'despike_' + workdirnme
        globstr = 'ds' + globstr
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')

    workdir, exists = utils.make_dir(subdir, workdirnme)

    if exists:
        logger.error('{0}: skipping {1} exists'.format(subdir, workdir))
        return None

    funcs = split(rawdir, destdir, sid, logger, globstr)
    if funcs is None:
        logger.error('Raw dir missing data: {0}'.format(rawdir))
        return None

    staligned, move_arr = make_realignst(funcs, tr, logger)
    if staligned is None:
        return None

    ## Make MEAN
    meanaligned = os.path.join(workdir, 'meanalign_{0}.nii.gz'.format(sid))
    meanaligned = utils.make_mean(staligned, 'meanalign_'.format(sid))
    logger.info(meanaligned)

    ## Make aligned_4d
    aligned4d = os.path.join(workdir, 'align4d_{0}.nii.gz'.format(sid))
    aligned4d = utils.fsl_make4d(staligned, aligned4d)
    logger.info(aligned4d)
    ## spm only has 6 params, add empty 7th for plotting
    move_arr = np.hstack((move_arr,
                          np.zeros((move_arr.shape[0],1))))
    plot_write_movement(workdir, sid, move_arr, logger)
    logger.info('{0} : finished'.format(sid))



def process_all(datadir, globstr, tr, logger, despike=False):
    gstr = os.path.join(datadir, globstr)
    subjects = sorted(glob(gstr))
    if len(subjects) < 1:
        raise IOError('{0}: returns no subjects'.format(gstr))
    for subjectdir in subjects:
        logger.info(subjectdir)
        res = process_subject(subjectdir, tr, logger, despike=despike)
        #res  = None
        if res is None:
            logger.error('{0}: skipped'.format(subjectdir))
            continue
        logger.info('{0}: finished {1}'.format(subjectdir, __file__))



if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='Run subject through realign, slicetime (optional despike)')
    parser.add_argument(
        'datadir',
        type=str,
        help='directory holding data')
    parser.add_argument(
        'TR',
        type=float,
        help='(float) Repetition Time (TR) of timeseries data')
    parser.add_argument(
        '-d', '--despike',
        dest='despike',
        action='store_true'
        )
    try:
        args = parser.parse_args()
        print args
    except:
        parser.print_help()

    """

    logger = logging.getLogger('rsfmri')
    logger.setLevel(logging.DEBUG)
    ts = reg.timestr()
    fname = os.path.split(__file__)[-1].replace('.py', '')
    logfile = os.path.join(datadir,
                           'logs',
                           '{0}_logger_{1}.log'.format(fname, ts))
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(os.getenv('USER'))
    process_all(datadir, 'B13*', repetition_time, logger, despike)
    """
