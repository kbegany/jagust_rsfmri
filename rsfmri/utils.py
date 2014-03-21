import os, sys
import datetime
from glob import glob
import json

import numpy as np

from skimage.morphology import binary_erosion
from nitime.timeseries import TimeSeries
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer

import nibabel
import nipype.interfaces.spm as spm
from nipype.interfaces.base import CommandLine
import nipype.interfaces.fsl as fsl
from nipype.utils import filemanip
import nipype.interfaces.afni as afni

## deal with relative import for now
cwd = os.getcwd()
sys.path.insert(0, cwd)
import nipype_ext

########################

## naming structure used in scripts to make subdirectories
defaults = {
    'rawdir': 'raw',
    'func_glob': 'B*func4d.nii*',
    'despiked_func_glob' : 'dsB*func4d.nii*',
    'anat_glob' : 'brainmask.nii*',
    'aparc_glob' : 'aparcaseg.nii*',
    'aligned' : 'align4d_{0}.nii*',
    'realign_ants':'ants_realign',
    'realign_spm': 'spm_realign_slicetime',
    'despike' : 'despike_',
    'coreg' : 'coreg_masks',
    'bandpass' : 'bandpass',
    'model_fsl': 'model_fsl',
    'wm_labels': [2,41, 77,78,79],
    'vent_labels': [4,5,14,15,28,43,44,60,72,75,76]
    }

def get_files(dir, globstr):
    """
    uses glob to find dir/globstr
    returns sorted list; number of files
    """
    searchstr = os.path.join(dir, globstr)
    files = glob(searchstr)
    files.sort()
    return files, len(files)


def make_datestr():
    now = datetime.datetime.now()
    return now.strftime('%Y_%m_%d_%H_%S')



def make_dir(base_dir, dirname='newdir'):
    """ makes a new directory if it doesnt alread exist
    returns full path

    Parameters
    ----------
    base_dir : str
    the root directory
    dirname  : str (default pib_nifti)
    new directory name

    Returns
    -------
    newdir  : str
    full path of new directory
    """
    newdir = os.path.join(base_dir,dirname)
    if not os.path.isdir(base_dir):
        raise IOError('ERROR: base dir %s DOES NOT EXIST'%(base_dir))
    directory_exists = os.path.isdir(newdir)
    if not directory_exists:
        os.mkdir(newdir)
    return newdir, directory_exists




def fsl_make4d(infiles, newfile):
    """a list of files is passed, a 4D volume will be created
    in the same directory as the original files"""
    if not hasattr(infiles, '__iter__'):
        raise IOError('expected list,not %s'%(infiles))
    startdir = os.getcwd()
    pth, nme = os.path.split(infiles[0])
    os.chdir(pth)
    merge = fsl.Merge()
    merge.inputs.in_files = infiles
    merge.inputs.dimension = 't'
    merge.inputs.merged_file = newfile
    out = merge.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None
    else:
        return out.outputs.merged_file

def fsl_split4d(infile, outdir, sid):
    """ uses fsl to split 4d file into parts
    based on sid, puts resulting files in outdir
    """
    startdir = os.getcwd()
    pth, nme = os.path.split(infile)
    os.chdir(outdir)
    im = fsl.Split()
    im.inputs.in_file = infile
    im.inputs.dimension = 't'
    im.inputs.out_base_name = sid
    im.inputs.output_type = 'NIFTI'
    out = im.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None
    else:
        # fsl split may include input file as an output
        ## bad globbing...
        # remove it here
        outfiles = out.outputs.out_files
        outfiles = [x for x in outfiles if not x == im.inputs.in_file]
        return outfiles



def get_slicetime(nslices):
    """
    If TOTAL # SLICES = EVEN, then the excitation order when interleaved
    is EVENS first, ODDS second.
    If TOTAL # SLICES = ODD, then the excitation order when interleaved is
    ODDS first, EVENS second.

    Returns:
    sliceorder: list
        list containing the order of slice acquisition used for slicetime
        correction

    """
    if np.mod(nslices,2) == 0:
        sliceorder = np.concatenate((np.arange(2,nslices+1,2),
                                     np.arange(1,nslices+1,2)))
    else:
        sliceorder = np.concatenate((np.arange(1,nslices+1,2),
                                     np.arange(2,nslices+1,2)))
    # cast to a list for use with interface
    return list(sliceorder)


def get_slicetime_vars(infiles, TR=None):
    """
    uses nibabel to get slicetime variables
    Returns:
    dict: dict
        nsclies : number of slices
        TA : acquisition Time
        TR: repetition Time
        sliceorder : array with slice order to run slicetime correction
    """
    if hasattr(infiles, '__iter__'):
        img = nibabel.load(infiles[0])
    else:
        img = nibabel.load(infiles)
    hdr = img.get_header()
    if TR is None:
        raise RuntimeError('TR is not defined ')
    shape = img.get_shape()
    nslices = shape[2]
    TA = TR - TR/nslices
    sliceorder = get_slicetime(nslices)
    return dict(nslices=nslices,
                TA = TA,
                TR = TR,
                sliceorder = sliceorder)


def save_json(inobj, outfile):
    ''' save inobj to outfile using json'''
    try:
        json.dump(inobj, open(outfile,'w+'))
    except:
        raise IOError('Unable to save %s to %s (json)'%(inobj, outfile))


def load_json(infile):
    ''' use json to load objects in json file'''
    try:
        result = json.load(open(infile))
    except:
        raise IOError('Unable to load %s' %infile)
    return result


def zip_files(files):
    if not hasattr(files, '__iter__'):
        files = [files]
    for f in files:
        base, ext = os.path.splitext(f)
        if 'gz' in ext:
            # file already gzipped
            continue
        cmd = CommandLine('gzip %s' % f)
        cout = cmd.run()
        if not cout.runtime.returncode == 0:
            logging.error('Failed to zip %s'%(f))


def unzip_file(infile):
    """ looks for gz  at end of file,
    unzips and returns unzipped filename"""
    base, ext = os.path.splitext(infile)
    if not ext == '.gz':
        return infile
    else:
        if os.path.isfile(base):
            return base
        cmd = CommandLine('gunzip %s' % infile)
        cout = cmd.run()
        if not cout.runtime.returncode == 0:
            print 'Failed to unzip %s'%(infile)
            return None
        else:
            return base


def afni_despike(in4d):
    """ uses afni despike to despike a 4D dataset
    saves as ds_<filename>"""
    dspike = afni.Despike()
    dspike.inputs.in_file = in4d
    dspike.inputs.outputtype = 'NIFTI_GZ'
    dspike.inputs.ignore_exception = True
    outfile = filemanip.fname_presuffix(in4d, 'ds')
    dspike.inputs.out_file = outfile
    res = dspike.run()
    return res.runtime.returncode, res


def spm_realign(infiles, matlab='matlab-spm8'):
    """ Uses SPM to realign files"""
    startdir = os.getcwd()
    pth, _ = os.path.split(infiles[0])
    os.chdir(pth)
    rlgn = spm.Realign(matlab_cmd = matlab)
    rlgn.inputs.in_files = infiles
    rlgn.inputs.register_to_mean = True
    out = rlgn.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None, None, None
    return out.outputs.mean_image, out.outputs.realigned_files,\
           out.outputs.realignment_parameters


def spm_realign_unwarp(infiles, matlab = 'matlab-spm8'):
    """ uses spm to run realign_unwarp
    Returns
    -------
    mean_img = File; mean generated by unwarp/realign

    realigned_files = Files; files unwarped and realigned

    parameters = File; file holding the trans rot params
    """

    startdir = os.getcwd()
    pth, _ = os.path.split(infiles[0])
    os.chdir(pth)
    ru = nipype_ext.RealignUnwarp(matlab_cmd = matlab)
    ru.inputs.in_files = infiles
    ruout = ru.run()
    os.chdir(startdir)
    if not ruout.runtime.returncode == 0:
        print ruout.runtime.stderr
        return None, None, None
    return ruout.outputs.mean_image, ruout.outputs.realigned_files,\
           ruout.outputs.realignment_parameters


def make_mean(niftilist, outfile):
    """given a list of nifti files
    generates a mean image"""
    if not hasattr(niftilist, '__iter__'):
        raise IOError('%s is not a list of valid nifti files,'\
            ' cannot make mean'%niftilist)
    n_images = len(niftilist)
    affine = nibabel.load(niftilist[0]).get_affine()
    shape =  nibabel.load(niftilist[0]).get_shape()
    newdat = np.zeros(shape)
    for item in niftilist:
        newdat += nibabel.load(item).get_data().copy()
    newdat = newdat / n_images
    newdat = np.nan_to_num(newdat)
    newimg = nibabel.Nifti1Image(newdat, affine)
    newimg.to_filename(outfile)
    return outfile

def mean_from4d(in4d, outfile):
    """ given a 4D files, calc mean across voxels (time)
    and write new 3D file to outfile with same mapping
    as in4d"""
    ##FIXME consider unzipping files first if this is slow
    ## Fast memmap doesnt work on zipped files very well
    affine = nibabel.load(in4d).get_affine()
    dat = nibabel.load(in4d).get_data()
    mean = dat.mean(axis=-1)
    newimg = nibabel.Nifti1Image(mean, affine)
    try:
        newimg.to_filename(outfile)
        return outfile
    except:
        raise IOError('Unable to write {0}'.format(outfile))

def simple_mask(dataf, maskf, outfile, thr=0):
    """ sets values in data to zero if they are zero in mask"""
    img = nibabel.load(dataf)
    dat = img.get_data()
    mask = nibabel.load(maskf).get_data()
    if not dat.shape == mask.shape:
        raise IOError('shape mismatch {0}, {1}'.format(dataf, maskf))
    dat[mask <=thr] = 0
    newimg = nibabel.Nifti1Image(dat, img.get_affine())
    newimg.to_filename(outfile)
    return outfile


def aparc_mask(aparc, labels, outfile = 'bin_labelmask.nii.gz'):
    """ takes coreg'd aparc and makes a mask based on label values
    Parameters
    ==========
    aparc : filename
        file containing label image (ints)
    labels : tuple
        tuple of label values (ints)
    """
    pth, _ = os.path.split(outfile)
    img = nibabel.load(aparc)
    mask = np.zeros(img.get_shape())
    label_dat = img.get_data()
    for label in labels:
        mask[label_dat == label] = 1
    masked_img = nibabel.Nifti1Image(mask, img.get_affine())
    outf = os.path.join(pth, outfile)
    masked_img.to_filename(outf)
    return outf

def erode(infile):
    """ use skimage.morphology to quickly erode binary mask"""
    img = nibabel.load(infile)
    dat = img.get_data().squeeze()
    kernel = np.zeros((3,3,3))
    kernel[1,:,:] = 1
    kernel[:,1,:] = 1
    kernel[:,:,1] = 1
    eroded = binary_erosion(dat, kernel)
    eroded = eroded.astype(int)
    newfile = filemanip.fname_presuffix(infile, 'e')
    newimg = nibabel.Nifti1Image(eroded, img.get_affine())
    newimg.to_filename(newfile)
    return newfile


def get_seedname(seedfile):
    _, nme, _ = filemanip.split_filename(seedfile)
    return nme


def extract_seed_ts(data, seeds):
    """ check shape match of data and seed if same assume registration
    extract mean of data in seed > 0"""
    data_dat = nibabel.load(data).get_data()
    meants = {}
    for seed in seeds:
        seednme = get_seedname(seed)
        seed_dat = nibabel.load(seed).get_data().squeeze()
        assert seed_dat.shape == data_dat.shape[:3]
        seed_dat[data_dat[:,:,:,0].squeeze() <=0] = 0
        tmp = data_dat[seed_dat > 0,:]
        meants.update({seednme:tmp.mean(0)})
    return meants


def bandpass_data():
    """ filters for 4D images and timeseries in txt files
    Uses afni 3dBandpass
    """
    pass

def nitime_bandpass(data, tr, ub=0.15, lb=0.0083):
    """ use nittime to bandpass filter regressors"""
    ts = TimeSeries(data, sampling_interval=tr)
    filtered_ts = FilterAnalyzer(ts, ub=ub, lb=lb)
    return filtered_ts.data

def write_filtered(data, outfile):
    data.to_file(outfile)


def bandpass_regressor():
    """ filters motion params and timeseries from csf and white matter
    (also global signal when relevant)
    Use afni  1dBandpass, motion values in a 1d file"""
    pass


def fsl_bandpass(infile, tr, lowf=0.0083, highf=0.15):
    """ use fslmaths to bandpass filter a 4d file"""
    startdir = os.getcwd()
    pth, nme = os.path.split(infile)
    os.chdir(pth)
    low_freq = 1  / lowf / 2 / tr
    high_freq = 1 / highf / 2 / tr
    im = fsl.ImageMaths()
    im.inputs.in_file = infile
    op_str = ' '.join(['-bptf',str(low_freq), str(high_freq)])
    im.inputs.op_string = op_str
    im.inputs.suffix = 'bpfilter_l%2.2f_h%2.2f'%(low_freq, high_freq)
    out = im.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None
    else:
        return out.outputs.out_file


def spm_slicetime(infiles, matlab_cmd='matlab-spm8',stdict = None):
    """
    runs slice timing
    returns
    timecorrected_files
    """
    startdir = os.getcwd()
    pth, _ = os.path.split(infiles[0])
    os.chdir(pth)
    if stdict == None:
        stdict = get_slicetime_vars(infiles)
    sliceorder = stdict['sliceorder']
    st = spm.SliceTiming(matlab_cmd = matlab_cmd)
    st.inputs.in_files = infiles
    st.inputs.ref_slice = np.round(stdict['nslices'] / 2.0).astype(int)
    st.inputs.slice_order = sliceorder
    st.inputs.time_acquisition = stdict['TA']
    st.inputs.time_repetition = stdict['TR']
    st.inputs.num_slices = stdict['nslices']
    out = st.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None
    else:
        return out.outputs.timecorrected_files


def spm_coregister(moving, target, apply_to_files=None,
                   matlab_cmd='matlab-spm8'):
    """
    runs coregistration for moving to target
    """
    startdir = os.getcwd()
    pth, _ = os.path.split(moving)
    os.chdir(pth)
    cr = spm.Coregister(matlab_cmd = matlab_cmd)
    cr.inputs.source = moving
    cr.inputs.target = target
    if apply_to_files is not None:
        cr.inputs.apply_to_files = apply_to_files
    out = cr.run()
    os.chdir(startdir)
    if not out.runtime.returncode == 0:
        print out.runtime.stderr
        return None, None
    else:
        return out.outputs.coregistered_source,\
               out.outputs.coregistered_files


def update_fsf(fsf, fsf_dict):
    """ update fsf with subject specific data
    Parameters
    ----------
    fsf : filename
        filename of default fsf file with default parameters
        to use for your model
    fsf_dict : dict
        dictionary holding data with the following keys:
        nuisance_dir
        nuisance_outdir
        input_data
        TR
        nTR

    Returns
    -------
    tmp5 : string
        string to write to new fsf file
    """
    original = open(fsf).read()
    tmp1 = original.replace('nuisance_dir',
                            fsf_dict['nuisance_dir'])
    tmp2 = tmp1.replace('nuisance_model_outputdir',
                        fsf_dict['nuisance_outdir'])
    tmp3 = tmp2.replace('nuisance_model_input_data',
                        fsf_dict['input_data'])
    tmp4 = tmp3.replace('nuisance_model_TR',
                        fsf_dict['TR'])
    tmp5 = tmp4.replace('nuisance_model_numTRs',
                        fsf_dict['nTR'])
    return tmp5

def write_fsf(fsf_string, outfile):
    """ writes an updated fsf string (see update_fsf)
    to outfile"""
    with open(outfile, 'w+') as fid:
        fid.write(fsf_string)


def run_feat_model(fsf_file):
    """ runs FSL's feat_model which uses the fsf file to generate
    files necessary to run film_gls to fit design matrix to timeseries"""
    clean_fsf = fsf_file.strip('.fsf')
    cmd = 'feat_model %s'%(clean_fsf)
    out = pp.CommandLine(cmd).run()
    if not out.runtime.returncode == 0:
        return None, out.runtime.stderr

    mat = fsf_file.replace('.fsf', '.mat')
    return mat, cmd


def run_film(data, design, results_dir):
    minval = nibabel.load(data).get_data().min()
    film = fsl.FILMGLS()
    film.inputs.in_file = data
    film.inputs.design_file = design
    film.inputs.threshold = minval
    film.inputs.results_dir = results_dir
    film.inputs.smooth_autocorr = True
    film.inputs.mask_size = 5
    res = film.run()
    return res
