import os, sys, re
import nibabel
import argparse
from glob import glob

## import utils
sys.path.insert(0,'/home/jagust/cindeem/CODE/jagust_rsfmri/rsfmri')
import utils



def get_tr(funcf):
    '''Uses the header of the functional image to find the time-step (TR)
    INPUT
        funcf : str (filename of a functional MRI image in nifti format)
    OUTPUT
        TR : str
    '''
    dat = nibabel.load(funcf)
    hdr = dat.get_header()
    raw_hdr = hdr.structarr.tolist()
    for metadat in raw_hdr:
        if 'TR:' in str(metadat):
            searchstr = '[1,2]{1}[0-9]{2}0.000'
            g = re.search(searchstr, metadat)
            tr = g.group()
            return float(tr)/1000

def run_slicetime_correction(datadir, despike):
    globstr = '%s/B*/raw/B*func4d.nii.gz' % (datadir)
    if despike:
        globstr = 'ds' + globstr
    funcfs = sorted(glob(globstr))
    for funcf in funcfs:
        raw_dir, _ = os.path.split(funcf)
        _, subsess = os.path.split(raw_dir)
        tr = get_tr(funcf)
        stdict = utils.get_slicetime_vars(infiles=funcf, TR=tr)
        st_func = utils.spm_slicetime(infiles=funcf, stdict=stdict)
        if despike:
            fname = '%s/dsst%s_func4d.nii.gz' % (raw_dir, subsess)
        else:
            fname = '%s/st%s_func4d.nii.gz' % (raw_dir, subsess)
        utils.fsl_make4d(st_func, fname)



if __name__ == '__main__':

    epilog = """
    python slicetime.py /home/jagust/graph/data/mri1.5/rest -d
    """
    parser = argparse.ArgumentParser(
        epilog = epilog,
        description='Run subject through slicetime (optional despike)')
    parser.add_argument(
        'datadir',
        type=str,
        help='directory holding data')
    parser.add_argument(
        '-d', '--despike',
        dest='despike',
        action='store_true'
        )
    try:
        args = parser.parse_args()
        print args
    except:
        parser.print_help()
        sys.exit()

    datadir = args.datadir
    despike = args.despike
    
    run_slicetime_correction(datadir, despike)
