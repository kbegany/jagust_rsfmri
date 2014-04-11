import os
import sys
import argparse
import logging
from glob import glob

import pandas

from rsfmri import utils
from rsfmri import register as reg

def find_make_workdir(subdir, despike, spm, logger=None):
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
    if exists:
        if logger:
            logger.error('{0}: skipping {1}  existS'.format(subdir, bpdir))
        raise IOError('{0}: EXISTS, Skipping'.format(bpdir))
    return rlgn_dir, workdir, bpdir

def setup_logging(workdir, sid):
    logger = logging.getLogger('bandpass')
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


def get_regressors(rlgndir):
    """open xml doc and return an array
    nregressors X ntimepoints"""
    xls = get_file(rlgndir, "B*movement.xls")
    ef = pandas.ExcelFile(xls)
    df = ef.parse(ef.sheet_names[0])
    ## ANTS movement missing padded 0's
    if 'ants' in rlgndir:
        df = utils.zero_pad_movement(df)
    # data is ntimepoints X nregressors
    # transpose and drop 7th item
    return df.values[:,:-1].T


def process_subject(subdir, tr, despike=False, spm=False):

    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    rlgndir, coregdir, bpdir = find_make_workdir(subdir, despike, spm)
    logger = setup_logging(bpdir, sid)
    globtype = utils.defaults['aligned'].format(sid)
    aligned = get_file(rlgndir, globtype, logger)
    # regressors are a
    regressors = get_regressors(rlgndir)
    # bandpass data and movement regressors
    bpaligned = utils.filemanip.fname_presuffix(aligned, prefix='bp',
        newpath=bpdir)
    logger.info(bpaligned)
    outfile = utils.fsl_bandpass(aligned, bpaligned, tr)
    if not outfile:
        logger.error('{}: bandpass failed'.format(aligned))
        raise IOError('{}: bandpass failed'.format(aligned))
    bpregressors = utils.nitime_bandpass(regressors, tr)
    movefiles = [os.path.join(bpdir, x) for x in utils.defaults['movement_names']]
    for dat, outfile in zip(bpregressors, movefiles):
        dat.tofile(outfile, sep='\n')
        logger.info(outfile)
    # extract WM, GM and global
    wm = get_file(coregdir, 'eB*WM_mask.nii*', logger)
    vent = get_file(coregdir,'B*VENT_mask.nii.gz', logger)
    aparc = get_file(coregdir, 'invxfm_aparcaseg.nii*', logger)

    noisefiles = [os.path.join(bpdir, x) for x in utils.defaults['noise_names']]
    seedlist = utils.extract_seed_ts(bpaligned, (wm,vent,aparc))
    for seed, outfile in zip(seedlist, noisefiles):
        demeaned = seed - seed.mean()
        demeaned.tofile(outfile, sep='\n')
        logger.info(outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Bandpass 4D data, handle regressors')
    parser.add_argument(
        'datadir',
        type=str,
        help='directory holding data')
    parser.add_argument(
        'tr',
        type=float,
        help='TR : data Repetition Time')
    parser.add_argument(
        '-g',
        '--globstr',
        type=str,
        default='B*',
        help='Optional glob string to get data (B*)')
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
    process_subject(allsub[taskid], args.tr, args.despike, args.spm)
