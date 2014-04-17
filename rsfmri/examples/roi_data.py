import os
import sys
from glob import glob
import logging

import numpy as np
import pandas
import argparse

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
            logger.error('{0}: skipping {1}  exists'.format(subdir, bpdir))
        raise IOError('{0}: Missing, Skipping'.format(bpdir))
    modeldirnme = utils.defaults['model_fsl']
    if gsr: # global signal regression
        modeldirnme  = modeldirnme + '_gsr'
    modeldir, exists = utils.make_dir(bpdir, modeldirnme)
    if not exists:
        if logger:
            logger.error('{0}: skipping {1}  exists'.format(subdir, modeldir))

    adjmatdir, exists = utils.make_dir(modeldir, 'adjacency_matrix')
    if exists:
        if logger:
            logger.error('{0}: skipping {1}  exists'.format(subdir, adjmatdir))
        raise IOError('{0}: EXISTS, Skipping'.format(adjmatdir))
    return rlgn_dir, workdir, bpdir, modeldir, adjmatdir

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

def make_ts_df(csv, data, maskf):
    """Using labels, values in the csv file
    pull mean timeseries from data using labels
    in mask

    Returns
    -------

    timeseries : pandas.DataFrame
        nlabels X ntimepoints
    """
    df = pandas.read_csv(csv, header=None, sep=None)
    labels = df[1].values
    result = utils.mask4d_with3d(data, maskf, labels)
    timeseries = pandas.DataFrame(result, index = df[0])
    return timeseries

def run_subject(subdir, gsr=False, despike=False, spm=False):
    _, sid = os.path.split(subdir)
    csv = '/home/jagust/cindeem/CODE/petproc-stable/gui/roi_csvs/fs_gm.csv'
    (rlgndir, coregdir,
        bpdir, modeldir, adjmatdir) = find_make_workdir(subdir, despike,
        spm, gsr)
    logger = setup_logging(adjmatdir, sid)
    logger.info(adjmatdir)
    finaldir = os.path.join(modeldir, 'resids')
    res = os.path.join(modeldir, 'resids', 'res4d.nii.gz')
    aparc = os.path.join(coregdir, 'invxfm_aparcaseg.nii.gz')
    timeseries = make_ts_df(csv,res,aparc)
    outf = os.path.join(adjmatdir, 'timeseries.csv')
    timeseries.to_csv(outf)
    corr = timeseries.T.corr()
    outf = os.path.join(adjmatdir, 'adjmat.csv')
    corr.to_csv(outf)
    npcorr = corr.values
    np.save(outf.replace('.csv','.npy'), npcorr)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract Freesurfer regions and create adjmat')
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
    run_subject(allsub[taskid], args.gsr,
        args.despike, args.spm)



