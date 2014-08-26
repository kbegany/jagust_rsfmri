import sys, re, os

import numpy as np
import pandas as pd
from glob import glob

from nipype.utils import filemanip
from nipype.utils.filemanip import (fname_presuffix, copyfile, split_filename)

import utils
import nipype_ext as npe




def get_age_dict():
    # Get age info
    subxlsx = '/home/jagust/graph/data/Renaud_Restingstate_subjs_11_19_13-1.xlsx'
    subdat = pd.ExcelFile(subxlsx).parse('all-sub_all-timepoints')
    agedict = dict(zip(subdat.LBNLID.values, subdat.age.values))
    return agedict


def get_files_old_only(datadir, searchstr):
    age_dict = get_age_dict()
    # Load data
    searchstr = '%s/%s' % (datadir, searchstr)
    files = sorted(glob(searchstr))
    # Remove young files
    for f in files:
        # Get subid
        subidpat = 'B[0-9]{2}-[0-9]{3}_[a,b,c]{1}'
        m = re.search(subidpat, f)
        subid = m.group()
        # remove if young
        if age_dict[subid] < 60:
            files.remove(f)
    return files


def spm_vbm8(infile):
    startdir = os.getcwd()
    pth, _ = os.path.split(infile)
    os.chdir(pth)
    vbm = npe.VBM8(matlab_cmd = 'matlab-spm8')
    vbm.inputs.in_files =[infile]
    vbm.inputs.write_warps = [True,True]
    vbm.inputs.write_gm = [1,0, 0,2] #write native, dartel afffine
    vbm.inputs.write_wm = [1,0, 0,2] #write native, dartel afffine
    vbm.inputs.write_csf = [1,0, 0,2] #write native, dartel afffine
    #vbm.inputs.write_bias = [1,0,0,2]
    vbmout = vbm.run()
    os.chdir(startdir)
    return vbmout
   

def get_vbmfiles(infile):
    """given the original T1  vbm was run on...
    find the expected output files as a dict
    T1
    gm_native
    wm_native
    csf_native
    gm_dartel
    wm_dartel
    icv_file
    icv (from *_seg8.txt
    """
    outdict = dict(T1=None,
                   gm_native = None,
                   wm_native = None,
                   csf_native = None,
                   gm_dartel = None,
                   wm_dartel = None,
                   icv_file = None,
                   icv = None)
    outdict['T1'] = infile
    outdict['gm_native'] = fname_presuffix(infile, prefix='p1')
    outdict['gm_dartel'] = fname_presuffix(infile, prefix='rp1', suffix = '_affine')
    outdict['wm_native'] = fname_presuffix(infile, prefix='p2')
    outdict['wm_dartel'] = fname_presuffix(infile, prefix='rp2', suffix = '_affine')
    outdict['csf_native'] = fname_presuffix(infile, prefix='p3')
    outdict['icv_file'] = fname_presuffix(infile, prefix='p', suffix='_seg8.txt',use_ext=False)
    #check for existence
    for item in [x for x in outdict if not outdict[x]==None]:
        if not os.path.isfile(outdict[item]):
            print item, outdict[item], ' was NOT generated/found'
            outdict[item] = None
    if outdict['icv_file'] is not None:
        outdict['icv'] = np.loadtxt(outdict['icv_file']).sum()
    return outdict


def spm_dartel_make(gm, wm, template_dir, template_nme):
    """ run dartel to make template and flowfields
    template_dir will be location of saved templates
    template_nme will be used to name template and
    flow fields"""
    startdir = os.getcwd()
    os.chdir(template_dir)
    dartel = npe.DARTEL(matlab_cmd = 'matlab-spm8')
    dartel.inputs.image_files = [gm, wm]
    dartel.inputs.template_prefix = template_nme
    dartel_out = dartel.run()
    os.chdir(startdir)
    return dartel_out


def move_flowfields(inflowfields):
    flowfields = []
    for ff in inflowfields:  
        pth, nme, ext = split_filename(ff)
        subdir, _ = os.path.split(pth)
        darteldir,exists = make_dir(subdir, dirname='dartel')
        newff = copy_file(ff, darteldir)
        remove_files([ff])
        flowfields.append(newff)
    return flowfields


def write_dartel_log(templates, flowfields):
    """write a log to describe template"""
    pth, nme, ext = split_filename(templates[0])
    logfile = os.path.join(pth, nme + '.log')
    with open(logfile, 'w+') as fid:
        for t in templates:
            fid.write(t + '\n')
        fid.write('\n')
        for ff in flowfields:
            fid.write(ff + '\n')
    return logfile



if __name__ == '__main__':

    ### Change items here ##############################################
    # get structurals
    datadir = '/home/jagust/graph/data/mri1.5/tr220'
    anatstr = 'B*/raw/B*_anat.nii.gz'
    anatomicals = get_files_old_only(datadir, anatstr)
    ####################################################################

    # run vbm
    vbm8_dict = {}
    for anat in anatomicals:
        # copy file to processing dir
        pth, nme, ext = filemanip.split_filename(anat)
        subid = nme.split('_anat')[0]
        vbmdir, exists = utils.make_dir(pth, 'vbm8')
        canat = utils.copy_file(anat, vbmdir)
        if not exists: # only run if vbm directory is missing
            out = spm_vbm8(canat)
        vbmd = get_vbmfiles(canat)
        vbm8_dict.update({subid: vbmd})
    
    # run dartel on cohort
    
    gms = get_files_old_only(datadir, 'B*/raw/anat/vbm8/rp1*.nii')
    wms = get_files_old_only(datadir, 'B*/raw/anat/vbm8/rp2*.nii')
    gms.sort()
    wms.sort()
    files = []
    pth, nme, ext = filemanip.split_filename(gms[0])
    datestr = utils.make_datestr()
    tmplt_nme = 'dartel_%s'%(datestr)
    dout = spm_dartel_make(gms, wms, templatedir, tmplt_nme)
    
    template = get_files_old_only(datadir,'B*/anat/vbm8/%s*'%(tmplt_nme))

    templatedir, exists = utils.make_dir(datadir,'template')
    newtemplate = utils.copy_files(template, templatedir)
    utils.remove_files(template)
    flowfieldstmp = utils.get_files(datadir,'*/anat/vbm8/*%s*'%(tmplt_nme))
    flowfields = move_flowfields(flowfieldstmp)
    dartellog = write_dartel_log(newtemplate, flowfields)
