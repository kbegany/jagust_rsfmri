import os
from glob import glob
from nipype.interfaces.base import CommandLine


def find_fsfiles(subdir, fsdir, ftype='brainmask'):
    _, sid = os.path.split(subdir)
    niitype = ftype.replace('+','')
    nii = os.path.join(subdir, 'raw','{0}.nii*'.format(niitype))
    try:
        nii = glob(nii)[0]
        print sid, nii, 'exists'
        return True, None, nii
    except:
        nii = os.path.join(subdir, 'raw','{0}.nii.gz'.format(niitype))
    mgz = os.path.join(fsdir, sid, 'mri', '{0}.mgz'.format(ftype))
    if not os.path.isfile(mgz):
        print ftype, ' not found in ', fsdir, sid
        return False, None, None
    return False, mgz, nii

def convert(infile, outfile):
    """converts freesurfer .mgz format to nifti
    """
    c1 = CommandLine('mri_convert --out_orientation LAS %s %s'%(infile,
                                                                outfile))
    out = c1.run()
    if not out.runtime.returncode == 0:
        #failed
        print 'did not convert %s from .mgz to .nii'%(infile)
    else:
        path = os.path.split(infile)[0]
        niifile = os.path.join(path,outfile)
        return niifile


rsdir = '/home/jagust/graph/data/mri1.5/tr220'
fsdir = '/home/jagust/graph/data/mri1.5/freesurfer'

globstr = os.path.join(rsdir, 'B*')
allsub =  sorted(glob(globstr))

for sub in allsub:
    for ftype in ('brainmask', 'aparc+aseg'):
        print ftype, sub
        exists, mgz, nii = find_fsfiles(sub, fsdir, ftype)
        if (exists) or (mgz is  None):
            print sub, exists, mgz
            continue
        convert(mgz, nii)



