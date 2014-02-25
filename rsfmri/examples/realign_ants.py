import os
import sys
from glob import glob


from rsfmri import utils
from rsfmri import register as reg
from rsfmri import transform



def split(rawdir, destdir, sid):
    files, nfiles = utils.get_files(rawdir, 'B*func4d.nii*')
    if not nfiles == 1:
        raise IOError('raw functional not found unexpected {0}'.format(files))
    rawfunc = files[0]
    funcs =  utils.fsl_split4d(rawfunc, destdir, sid)
    if funcs is None:
        raise IOError('splitting {0} failed'.format(rawfunc))
    return funcs


def make_realign_splitfiles(rawdir, destdir, sid):
    """ grabs raw data, splits into destdir"""
    funcs = split(rawdir, destdir, sid)
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
    utils.zip_files(funcs)
    utils.zip_files(aligned[1:])
    movement_array = transform.collate_affines(xfms)
    return aligned, movement_array


def plot_write_movement(destdir, sid, movement):
    transform.plot_movement(movement, destdir)
    outfile = os.path.join(destdir, '{0}_movement.xls'.format(sid))
    transform.movementarr_to_pandas(movement, outfile)
  

def process_subject(subdir):
    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    workdir, exists = utils.make_dir(subdir, 'ants_realign')

    if exists:
        raise IOError('{0}: skipping {1} exists'.format(subdir, workdir))
        
    aligned, move_arr = make_realign_splitfiles(rawdir, 
                                                workdir, 
                                                sid) 
 
    plot_write_movement(workdir, sid, move_arr)
    print '{0} : finished'.format(sid)

if __name__ == '__main__':

    try:
        datadir = sys.argv[1]
    except:
        raise IOError("""no data directory defined:
            USAGE:
            python realignst_spm.py /path/to/data/subj
            """)



    process_subject(datadir)
    

    
    

