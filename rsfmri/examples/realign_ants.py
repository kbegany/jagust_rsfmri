import os
import sys
from glob import glob
import argparse

from rsfmri import utils
from rsfmri import register as reg
from rsfmri import transform



def split(rawdir, destdir, sid, despike=False):
    globstr = 'B*func4d.nii*'
    if despike:
        globstr = 'ds' + globstr

    files, nfiles = utils.get_files(rawdir, globstr)
    if not nfiles == 1:
        raise IOError('raw functional not found unexpected {0}'.format(globstr))
    rawfunc = files[0]
    funcs =  utils.fsl_split4d(rawfunc, destdir, sid)
    if funcs is None:
        raise IOError('splitting {0} failed'.format(rawfunc))
    return funcs


def make_realign(funcs):
    """ uses ANTS to realign files"""
    target = funcs[0]
    aligned = [target]
    xfms = []
    for moving in funcs[1:]:
        xfm = reg.affine_register_cc(target, moving)
        if xfm is None:
            raise IOError('{0} :affine register failed'.format(moving))
        xfms.append(xfm)
        transformed = reg.apply_transform(moving, xfm, target=target)
        if transformed is None:
            raise IOError('{0}: apply xfm failed'.format(moving))
        aligned.append(transformed)
    movement_array = transform.collate_affines(xfms)
    return aligned, movement_array


def plot_write_movement(destdir, sid, movement):
    transform.plot_movement(movement, destdir)
    outfile = os.path.join(destdir, '{0}_movement.xls'.format(sid))
    transform.movementarr_to_pandas(movement, outfile)


def process_subject(subdir, despike=False):
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    workdir = 'ants_realign'
    if despike:
        workdir = 'despike_' + workdir
    workdir, exists = utils.make_dir(subdir, workdir)

    if exists:
        raise IOError('{0}: skipping {1} exists'.format(subdir, workdir))

    funcs = split(rawdir, workdir, sid, despike)
    aligned, move_arr = make_realign(funcs)

    meanaligned = os.path.join(workdir, 'meanalign_{0}.nii.gz'.format(sid))
    meanaligned = utils.make_mean(aligned, 'meanalign_'.format(sid))
    print 'meanaligned is:',meanaligned

    ## Make aligned_4d
    aligned4d = os.path.join(workdir, 'align4d_{0}.nii.gz'.format(sid))
    aligned4d = utils.fsl_make4d(aligned, aligned4d)
    print 'aligned_4d:', aligned4d
    plot_write_movement(workdir, sid, move_arr)
    utils.zip_files(funcs)
    utils.zip_files(aligned[1:])
    print '{0} : finished'.format(sid)

def process_all(datadir, globstr, taskid, despike=False):
    globstr = os.path.join(datadir, globstr)
    allsub = sorted(glob(globstr))
    subdir = allsub[taskid]
    process_subject(subdir, despike)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run ANTS realign')
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
        action='store_true'
        )
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
    datadir, globstr, despike = args.datadir, args.globstr, args.despike
    process_all(datadir, globstr, taskid, despike)


