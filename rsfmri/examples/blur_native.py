import os
import sys
import argparse
import logging
from glob import glob

import pandas

from rsfmri import utils
from rsfmri import register as reg


def find_make_workdir(subdir, despike, spm, gsr=False, logger=None):
    """ generates realign directory to query based on flags
    and if it exists and create new workdir
    """
    rlgn_dir = utils.defaults['realign_ants']
    if spm:
        rlgn_dir = utils.defaults['realign_spm']
    if despike:
        rlgn_dir = utils.defaults['despike'] + rlgn_dir
    rlgn_dir = os.path.join(subdir, rlgn_dir)
    if not os.path.isdir(rlgn_dir):
        if logger:
            logger.error('{0} doesnt exist skipping'.format(rlgn_dir))
        raise IOError('{0} doesnt exist skipping'.format(rlgn_dir))
    if logger:
        logger.info(rlgn_dir)
    workdirnme = utils.defaults['coreg']
    workdir, exists = utils.make_dir(rlgn_dir, workdirnme)
    if not exists:
        if logger:
            logger.error('{0}: skipping {1} doesnt exist'.format(subdir, workdir))
        raise IOError('{0}: MISSING, Skipping'.format(workdir))
    bpdirnme = utils.defaults['bandpass']
    bpdir, exists = utils.make_dir(workdir, bpdirnme)
    if not exists:
        if logger:
            logger.error('{0}: skipping {1}  existS'.format(subdir, bpdir))
        raise IOError('{0}: Missing, Skipping'.format(bpdir))
    modeldirnme = utils.defaults['model_fsl']
    if gsr: # global signal regression
        modeldirnme  = modeldirnme + '_gsr'
    modeldir, exists = utils.make_dir(bpdir, modeldirnme)
    resids_dir, exists = utils.make_dir(modeldir, 'resids')
    if not exists:
        if logger:
            logger.error('{0}: skipping {1}  existS'.format(subdir, resids_dir))
        raise IOError('{0}: Missing, Skipping'.format(resids_dir))
    return resids_dir

def setup_logging(workdir, sid):
    logger = logging.getLogger('fsl_model')
    logger.setLevel(logging.DEBUG)
    ts = reg.timestr()
    fname = os.path.split(__file__)[-1].replace('.py', '')
    logfile = os.path.join(workdir,
                           '{0}_{1}_{2}.log'.format(sid, fname, ts))
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(__file__)
    logger.info(ts)
    logger.info(workdir)
    logger.info(os.getenv('USER'))
    return logger

def get_file(workdir, type_glob, logger=None):
    files, nfiles = utils.get_files(workdir, type_glob)
    ## type_glob eg 'align4d_{0}.nii.gz'.format(sid) from utils.defaults)
    if logger:
        logger.info(' glob ({0}) yields nfiles: {1}'.format(type_glob, nfiles))
    if nfiles == 1:
        return files[0]
    logger.error('{0} in {1} not found'.format(type_glob, workdir))
    raise IOError('{0} in {1} not found'.format(type_glob, workdir))


def run_subject(subdir, gsr=False, despike=False, spm=False, fwhm=8):
    _, sid = os.path.split(subdir)

    resids_dir = find_make_workdir(subdir, despike, spm, gsr)
    logger = setup_logging(resids_dir, sid)
    logger.info(resids_dir)

    resid = get_file(resids_dir, 'res4d*', logger)
    cmd, smoothed = utils.smooth_to_fwhm(resid, outfile = None, fwhm = fwhm)
    logger.info(cmd)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Blur residuals in Native Space')
    parser.add_argument(
        'datadir',
        type=str,
        help='directory holding data')
    parser.add_argument(
        '-g',
        '--globstr',
        type=str,
        default='B*',
        help='Optional glob string to get data (B*)')
    parser.add_argument(
        '-fwhm',
        type=str,
        default='8',
        help='FWHM to smooth data to')
    parser.add_argument(
        '-d', '--despike',
        dest='despike',
        action='store_true',
        help='running on despiked data? (default False)'
        )
    parser.add_argument(
        '-spm', '--spm-aligned',
        dest='spm',
        action='store_true',
        help='spm aligned data? (default False)')
    parser.add_argument(
        '-gsr', '--global-signal-regression',
        dest='gsr',
        action='store_true',
        help='Put global signal regression into model (default False')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    try:
        taskid = os.environ['SGE_TASK_ID']
    except KeyError:
        taskid = 1
        #raise RuntimeError('Expecting to run via SGE, exiting')
    # taskid starts at 1, so we need to subtract 1 to work with code
    taskid = int(taskid) -1
    fulldir = os.path.join(args.datadir, args.globstr)
    print fulldir
    allsub = sorted(glob(fulldir))
    run_subject(allsub[taskid], args.gsr, args.despike, args.spm, args.fwhm)


