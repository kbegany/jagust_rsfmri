import os
import sys
from glob import glob
import argparse

from rsfmri import utils
from rsfmri import register as reg

import logging

def get_raw(subdir, globstr, logger):
    files, nfiles = utils.get_files(subdir, globstr)
    logger.info('SUBID: {0}, nfiles: {1}'.format(subdir, nfiles))
    if not nfiles == 1:
        logger.error('Raw functional not found: {0}'.format(globstr))
        return None
    return files[0]

def run_despike(datadir, globstr, taskid):
    gstr = os.path.join(datadir, 'B*')
    subjects = sorted(glob(gstr))
    subdir = subjects[taskid]
    _, sid = os.path.split(subdir)
    rawdir, exists = utils.make_dir(subdir, 'raw')
    if not exists:
        return None
    files, nfiles = utils.get_files(rawdir, 'ds*func*nii*')
    if nfiles > 0:
        print files, 'exists skipping'
        return None
    logger = logging.getLogger('despike')
    logger.setLevel(logging.DEBUG)
    ts = reg.timestr()
    fname = os.path.split(__file__)[-1].replace('.py', '')
    logfile = os.path.join(rawdir,
                           '{0}_{1}_{2}.log'.format(sid, fname, ts))
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(ts)
    logger.info(__file__)
    logger.info(os.getenv('USER'))
    in4d = get_raw(rawdir, globstr, logger)
    if in4d is None:
        logger.error('in4d is None')
        return
    returncode, result = utils.afni_despike(in4d)
    if returncode == 0:
        logger.info('Despike finished')
        logger.info(result.outputs.out_file)
        return
    logger.error(result.runtime.stderr)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run despike')
    parser.add_argument(
        'datadir',
        type=str,
        help='directory holding data')
    parser.add_argument(
        '-g',
        '--globstr',
        type=str,
        default='B*func4d.nii*',
        help='Optional glob string to get raw data (B*func4d.nii*')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    try:
        taskid = os.environ['SGE_TASK_ID']
    except KeyError:
        #taskid = 1
        raise RuntimeError('Expecting to run via SGE, exiting')
    # taskid starts at 1, so we need to subtract 1 to work with code
    taskid = int(taskid) -1
    datadir, globstr = args.datadir, args.globstr
    run_despike(datadir, globstr, taskid)
