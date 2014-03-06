import os
import sys
import argparse


from rsfmri import utils
from rsfmri import register as reg

def find_make_workdir(subdir, despike, spm):
    """ generates realign directory to query based on flags
    and if it exists and create new workdir
    """
    rlgn_dir = utils.defaults['realign_ants']
    if spm:
        rlgn_dir = utils.defaults['realign_spm']
    if despike:
        rlgn_dir = utils.defaults['despike'] + rlgn_dir
    rlgn_dir = os.path.join(subdir, rlgn_dir)
    if not os.isdir(rlgn_dir):
        logger.error('{0} doesnt exist skipping'.format(rlgn_dir))
        raise IOError('{0} doesnt exist skipping'.format(rlgn_dir))
    logger.info(rlgn_dir)
    workdirnme = utils.defaults['coreg']
    workdir, exists = utils.make_dir(rlgn_dirdir, workdirnme)
    if exists:
        logger.error('{0}: skipping {1} exists'.format(subdir, workdir))
        raise IOError('{0}: EXISTS, Skipping'.format(workdir))
    return rlgn_dir, workdir


def get_file(workdir, type_glob):
    files, nfiles = utils.get_files(workdir, type_glob)
    ## type_glob eg 'align4d_{0}.nii.gz'.format(sid) from utils.defaults)
    logger.info(' glob ({0}) yields nfiles: {1}'.format(type_glob, nfiles))
    if nfiles == 1:
        return files[0]
    raise IOError('{0} in {1} not found'.format(type_glob, workdir))


def process_subject(subdir):
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    rlgndir, coregdir = find_make_workdir(subdir, despike, spm)
    globtype = utils.defaults['aligned'].format(sid)
    aligned = get_file(rlgndir, globtype)
    bmask = get_file(rawdir, utils.defaults['anat_glob'])
    aparc = get_file(rawdir, utils.defaults['aparc_glob'])
    # copy files to coregdir
    ## FIXME continue from here



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Coreg ANAT to realigned func, and use aparc to make masks')
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