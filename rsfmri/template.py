import utils
from nipype.utils import filemanip


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


    datadir = '/home/jagust/graph/data/spm_220'
    # change this!!
    anats = utils.get_files(datadir, 'B*/anat/B*_anat.nii')
    gms = utils.get_files(datadir, '*/anat/vbm8/rp1*.nii')
    gms.sort()
    wms = utils.get_files(datadir, '*/anat/vbm8/rp2*.nii')
    wms.sort()
    
    files = []
    pth, nme, ext = filemanip.split_filename(gms[0])
    templatedir = pth
    datestr = utils.make_datestr()
    tmplt_nme = 'dartel_%s'%(datestr)
    dout = spm_dartel_make(gms, wms, templatedir, tmplt_nme)

    template = utils.get_files(datadir,'*/anat/vbm8/%s*'%(tmplt_nme))

    templatedir, exists = utils.make_dir(datadir,'template')
    newtemplate = utils.copy_files(template, templatedir)
    utils.remove_files(template)
    flowfieldstmp = utils.get_files(datadir,'*/anat/vbm8/*%s*'%(tmplt_nme))
    flowfields = move_flowfields(flowfieldstmp)
    dartellog = write_dartel_log(newtemplate, flowfields)
