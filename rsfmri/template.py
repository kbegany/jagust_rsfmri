import sys, re, os

import numpy as np
import pandas as pd
from glob import glob

from nipype.utils import filemanip
from nipype.utils.filemanip import (fname_presuffix, copyfile, split_filename)


import utils
import nipype_ext as npe


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

    # run dartel on cohort
    
    gms = utils.get_files(datadir, 'B*/despike_ants_realign/coreg_masks/aparcaseg.nii.gz')
    wms = utils.get_files(datadir, 'B*/despike_ants_realign/coreg_masks/B*_WM_mask.nii.gz')
    gms.sort()
    wms.sort()
    files = []
    pth, nme, ext = filemanip.split_filename(gms[0])
    datestr = utils.make_datestr()
    tmplt_nme = 'dartel_%s'%(datestr)
    templatedir = '/home/jagust/graph/data/mri1.5/tr220/template'
    dout = spm_dartel_make(gms, wms, templatedir, tmplt_nme)
    
    #template = get_files_old_only(datadir,'B*/anat/vbm8/%s*'%(tmplt_nme))

    templatedir, exists = utils.make_dir(datadir,'template')
    newtemplate = utils.copy_files(template, templatedir)
    utils.remove_files(template)
    #flowfieldstmp = utils.get_files(datadir,'*/anat/vbm8/*%s*'%(tmplt_nme))
    flowfields = move_flowfields(flowfieldstmp)
    dartellog = write_dartel_log(newtemplate, flowfields)
