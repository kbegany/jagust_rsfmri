import os
import sys
from glob import glob


from rsfmri import utils
from rsfmri import register as reg


def make_dirs(sibdir):
    """ creates the realign, realign_slicetime
    subdirectories in subjects(sid)  dir
    returns ants, spm"""
    subdirs = ('ants_realign', 'spm_realign_slicetime')
    all = []
    for sdir in subdirs:
        tmp = os.path.join(siddir, sdir)
        all.append(tmp)
    ants, spm = all
    return ants, spm


def split(rawdir, destdir, sid):
    files, nfiles = utils.get_files(rawdir, 'B*anat.nii*')
    if not nfiles == 1:
        raise IOError('func file returns unexpected {0}'.format(files))
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
    return aligned, xfms


def make_realignst_splitfiles(rawdir, destdir, sid, TR):
    funcs = utils.split(rawdir, destdir, sid)
    stvars = utils.get_slicetime_vars(funcs, TR=TR) 
    meanfunc, realigned, params = utils.spm_realign_unwarp(funcs)
    if meanfunc is None:
        raise IOError('{0}: spm realignment failed'%(funcs))
    st_realigned = utils.spm_slicetime(realigned, stdict = stvars)
    if st_realigned is None:
        raise IOError('{0}: slice timing failed'.format(realigned))
    return st_realigned, params


    
    

def ants_realign():
    """ affine_register_cc
    make mean
    write movement"""
    pass

def spm_slicetime_realign():
    """ spm slicetime realign, make mean
    write movement"""
    pass



