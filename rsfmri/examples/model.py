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

    if exists:
        if logger:
            logger.error('{0}: skipping {1}  existS'.format(subdir, bpdir))
        raise IOError('{0}: EXISTS, Skipping'.format(modeldir))
    return rlgn_dir, workdir, bpdir, modeldir

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

def make_fsf_dict(modeldir, data, tr, logger):
    try:
        ntrs = utils.nibabel.load(data).get_shape()[3]
    except:
        logger.error('cannot get time dim of {}'.format(bpaligned))
        raise IOError('cannot get time dim of {}'.format(bpaligned))
    nuisdir, _ = os.path.split(modeldir)
    outd = {
        'nuisance_dir' : nuisdir,
        'nuisance_outdir' : modeldir,
        'input_data' : data,
        'TR' : '{:2.2f}'.format(tr),
        'nTR' : '{:d}'.format(ntrs)
        }
    logger.info(outd)
    return outd

def get_fsf(gsr):
    tmpdir, _ = os.path.split(utils.__file__)
    if gsr:
        return os.path.join(tmpdir, 'nuisance.fsf')
    return os.path.join(tmpdir, 'nuisance_nogsr.fsf')

def run_subject(subdir, tr, gsr=False, despike=False, spm=False):
    _, sid = os.path.split(subdir)

    rlgndir, coregdir, bpdir, modeldir = find_make_workdir(subdir, despike,
        spm, gsr)
    logger = setup_logging(bpdir, sid)
    logger.info(modeldir)
    fsf = get_fsf(gsr)
    logger.info(fsf)
    logger.info('TR: {}'.format(tr))
    bpaligned = get_file(bpdir, 'bpalign4d*', logger)
    fsfd = make_fsf_dict(modeldir, bpaligned, tr, logger)
    fsffile = os.path.join(modeldir, '{}_nuisance.fsf'.format(sid))
    tmpfsf = utils.update_fsf(fsf, fsfd)
    utils.write_fsf(tmpfsf, fsffile)
    logger.info(fsffile)
    mat, cmd = utils.run_feat_model(fsffile)
    logger.info(cmd)
    finaldir = os.path.join(modeldir, 'resids')
    res = utils.run_film(bpaligned, mat, finaldir)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='FSL model, calc regressors')
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
    run_subject(allsub[taskid], args.tr,
        args.gsr, args.despike, args.spm)


