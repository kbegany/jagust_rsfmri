import os
import sys
from glob import glob
import argparse


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
    if exists:
        #logger.error('{0}: skipping {1} exists'.format(subdir, workdir))
        raise IOError('{0}: EXISTS, Skipping'.format(workdir))
    return rlgn_dir, workdir


def get_file(workdir, type_glob, logger=None):
    files, nfiles = utils.get_files(workdir, type_glob)
    ## type_glob eg 'align4d_{0}.nii.gz'.format(sid) from utils.defaults)
    if logger:
        logger.info(' glob ({0}) yields nfiles: {1}'.format(type_glob, nfiles))
    if nfiles == 1:
        return files[0]
    raise IOError('{0} in {1} not found'.format(type_glob, workdir))


def copy_file(infile, dest):
    cmd = 'cp {0} {1}'.format(infile, dest)
    os.system(cmd)
    _, nme = os.path.split(infile)
    return os.path.join(dest, nme)

def process_subject(subdir, despike=False, spm=False):
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    rlgndir, coregdir = find_make_workdir(subdir, despike, spm)
    globtype = utils.defaults['aligned'].format(sid)
    aligned = get_file(rlgndir, globtype)
    bmask = get_file(rawdir, utils.defaults['anat_glob'])
    aparc = get_file(rawdir, utils.defaults['aparc_glob'])
    # copy files to coregdir
    bmask = copy_file(bmask, coregdir)
    aparc = copy_file(aparc, coregdir)
    ## make mean
    mean_aligned = os.path.join(coregdir,
        '{0}_meanaligned.nii.gz'.format(sid))
    mean_aligned = utils.mean_from4d(aligned, mean_aligned)
    ## register mean to bmask
    xfm = reg.affine_register_mi(bmask, mean_aligned)
    ## invert and apply to brainmask and aparc
    if xfm is None:
        raise IOError('{0}: meanepi -> anat failed')
    transform = '-i {0}'.format(xfm)
    rbmask = reg.apply_transform(bmask, transform, target=mean_aligned)
    raparc = reg.apply_transform(aparc, transform,
        target=mean_aligned, use_nn=True)
    ## make masks
    wm_labels = utils.defaults['wm_labels']
    vent_labels = utils.defaults['vent_labels']
    wm_maskf = os.path.join(coregdir, '{0}_WM_mask.nii.gz'.format(sid))
    wm_mask = utils.aparc_mask(raparc, wm_labels, wm_maskf)
    vent_maskf = os.path.join(coregdir, '{0}_VENT_mask.nii.gz'.format(sid))
    vent_mask = utils.aparc_mask(raparc, vent_labels, vent_maskf)
    eroded_wm = utils.erode(wm_mask)
    eroded_vent = utils.erode(vent_mask)



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
    fulldir = os.path.join(args.datadir, args.globstr)
    allsub = sorted(glob(fulldir))
    process_subject(allsub[taskid], args.despike, args.spm)