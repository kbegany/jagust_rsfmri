import os
import sys
from glob import glob

import numpy as np

from rsfmri import utils
from rsfmri import register as reg
from rsfmri import transform

import logging


def split(rawdir, destdir, sid, logger):
    """ split files into 3D vols, 
    return is None if it fails"""
    globstr = 'B*func4d.nii*'
    files, nfiles = utils.get_files(rawdir, globstr)
    print sid, nfiles
    if not nfiles == 1:
        logger.error('Raw functional not found: {0}'.format(globstr))
        return None
    rawfunc = files[0]
    funcs =  utils.fsl_split4d(rawfunc, destdir, sid)
    return funcs


def make_realignst_splitfiles(rawdir, destdir, sid, TR, logger):
    funcs = split(rawdir, destdir, sid, logger)
    if funcs is None:
        logger.error('Raw dir missing data: {0}'.format(rawdir))
        return None, None
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
    


def process_subject(subdir, tr, logger):
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    workdir, exists = utils.make_dir(subdir, 'spm_realign_slicetime')

    if exists:
        logger.error('{0}: skipping {1} exists'.format(subdir, workdir))
        return None

    staligned, move_arr = make_realignst_splitfiles(rawdir, 
                                                    workdir, 
                                                    sid, 
                                                    tr, 
                                                    logger)
    if staligned is None:
        return None
    ## spm only has 6 params, add empty 7th for plotting
    move_arr = np.hstack((move_arr, 
                          np.zeros((move_arr.shape[0],1))))
    plot_write_movement(workdir, sid, move_arr, logger)
    logger.info('{0} : finished'.format(sid))



def process_all(datadir, globstr, tr, logger):
    gstr = os.path.join(datadir, globstr)
    subjects = sorted(glob(gstr))
    if len(subjects) < 1:
        raise IOError('{0}: returns no subjects'.format(gstr))
    for subjectdir in subjects:
        logger.info(subjectdir)
        res = process_subject(subjectdir, tr, logger)
        #res  = None
        if res is None:
            logger.error('{0}: skipped'.format(subjectdir))
            continue
        logger.info('{0}: finished {1}'.format(subjectdir, __file__))



if __name__ == '__main__':

    try:
        datadir = sys.argv[1]
    except:
        raise IOError("""no data directory defined:
            USAGE:
            python realignst_spm.py /path/to/data TR
            """)

    try:
        # need tr
        repetition_time = float(sys.argv[2])
    except:
        raise IOError("""no TR (repetition time) defined:
            USAGE:
            python realignst_spm.py /path/to/data TR
            """)        


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
    process_all(datadir, 'B13*', repetition_time, logger)

