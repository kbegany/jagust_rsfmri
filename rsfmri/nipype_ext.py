# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""SPM wrappers for preprocessing data

   Change directory to provide relative paths for doctests
   >>> import os
   >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
   >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
   >>> os.chdir(datadir)
"""

__docformat__ = 'restructuredtext'

# Standard library imports
from copy import deepcopy
import os
from glob import glob

# Third-party imports
import numpy as np

# Local imports
from nipype.interfaces.base import (OutputMultiPath, TraitedSpec, isdefined,
                                    traits, InputMultiPath, File)
from nipype.interfaces.spm.base import (SPMCommand, scans_for_fname,
                                        func_is_3d,
                                        scans_for_fnames, SPMCommandInputSpec)
from nipype.utils.filemanip import (fname_presuffix, filename_to_list,
                                    list_to_filename, split_filename)


class RealignUnwarpInputSpec(SPMCommandInputSpec):
    in_files = traits.Either(traits.List(File(exists=True)),File(exists=True),
                             field='data.scans',
                             mandatory=True,
                             desc='list of filenames to realign/uwarp', copyfile=True)
    phase_map =  File(exists=True, field='data.pmscan',
                      desc='phase map in alignment with first session, if empty, not run')                      
    #jobtype = traits.Enum('estwrite', 'estimate', 'write',
    #                      desc='one of: estimate, write, estwrite',
    #                      usedefault=True)
    quality = traits.Range(low=0.0, high=1.0, field='eoptions.quality',
                           desc='0.1 = fast, 1.0 = precise')
    fwhm = traits.Range(low=0.0, field='eoptions.fwhm',
                        desc='gaussian smoothing kernel width')
    separation = traits.Range(low=0.0, field='eoptions.sep',
                              desc='sampling separation in mm')
    register_to_mean = traits.Bool(field='eoptions.rtm',
                desc='Indicate whether realignment is done to the mean image')
    weight_img = File(exists=True, field='eoptions.weight',
                             desc='filename of weighting image')
    interp = traits.Range(low=0, high=7, field='eoptions.einterp',
                          desc='degree of b-spline used for interpolation')
    wrap = traits.List(traits.Int(), minlen=3, maxlen=3,
                        field='eoptions.ewrap',
                        desc='Check if interpolation should wrap in [x,y,z]')
    basis_functions = traits.List(traits.Int(), minlen=2, maxlen=2,
                                  field='uweoptions.basfcn',
                                  desc='Number of basis functions for each dimension default=[12,12]')
    regularisation_order = traits.Range(low=0, high=3, field='uweoptions.regorder',
                                        desc='Weight to balance compromise between likelihood function'\
                                        'and smoothing constraint')
    jacobian_deformations = traits.Bool(field='uweoptions.jm',
                                        desc='do jacobian intensity modulation, default is False')
    first_order_effects = traits.List(traits.Int(),maxlen=6,
                                      desc = 'Vector of first order effects to model; 1=xtran,2=ytran'\
                                      '3=ztran, 4=xrot, 5=yrot, 6=zrot, (default=[4,5], pitch and roll)')
    second_order_effects = traits.List(traits.Int(),maxlen=6,
                                      desc = 'Vector of second order effects to model; 1=xtran,2=ytran'\
                                      '3=ztran, 4=xrot, 5=yrot, 6=zrot, (default=[], None)')
    uwfwhm = traits.Range(low=0.0, field='uweoptions.uwfwhm',
                        desc='smoothing for unwarp (default = 4)')
    re_estimate_move = traits.Bool(field='uweoptions.rem',
                                   desc = 're-estimate movement parameters at each unwarp (Default = True)')
    num_iterations = traits.Range(low=1, high = 10, field='uweoptions.noi',
                                  desc = 'Number of iterations, (Default = 5)')
    taylor_expansion_point = traits.Enum('Average', 'First', 'Last',
                                         desc='time point to do taylor expansion around,'\
                                         '(default=Average; should give best variance reduction')
    


                                         
    write_which = traits.Tuple(traits.Int, traits.Int, field='uwroptions.uwwhich',
                              desc='determines which images to reslice '\
                               'default = (2,1) is all images plus mean')
    write_interp = traits.Range(low=0, high=7, field='uwroptions.rinterp',
                                desc='degree of b-spline used for interpolation'\
                                '(default= 4; 4th degree bspline), 0=nearestneighbor,1=trilinear')
    write_wrap = traits.List(traits.Int(), minlen=3, maxlen=3,
                              field='uwroptions.wrap',
                             desc='default = [0,0,0], Check if interpolation should wrap in [x,y,z]')
    write_mask = traits.Bool(field='uwroptions.mask',
                             desc='True/False mask output image (Default = True) ')

    write_prefix = traits.Str(default_value='u', field='uwroptions.prefix',
                              desc='string prepended to filenames of smoothed images, (Default = "u")')

class RealignUnwarpOutputSpec(TraitedSpec):
    mean_image = File(exists=True, desc='Mean image file from the unwarp/realign')
    realigned_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)),File(exists=True)),
                                      desc='Unwarped/Realigned files')
    realignment_parameters = OutputMultiPath(File(exists=True),
                    desc='Estimated translation and rotation parameters')

class RealignUnwarp(SPMCommand):
    """Use spm_realign for estimating within modality rigid body alignment
       and unwarping (an attempt to of remove movement related variance ) in a timeseries

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=25

    Examples
    --------

    >>> import nipype.interfaces.spm as spm
    >>> realignunwarp = spm.RealignUnwarp()
    >>> realignwarp.inputs.in_files = 'functional.nii'
    >>> realignwarp.inputs.register_to_mean = True
    >>> realignwarp.run() # doctest: +SKIP

    """

    input_spec = RealignUnwarpInputSpec
    output_spec = RealignUnwarpOutputSpec

    _jobtype = 'spatial'
    _jobname = 'realignunwarp'

    def _run_interface(self, runtime):
        """Executes the SPM function using MATLAB."""
        mscript = self._make_matlab_command(deepcopy(self._parse_inputs()))
        self.mlab.inputs.script = self.fix_mscript(mscript)
        #self.mlab.script = self.fix_mscript(mscript)
        print self.mlab.inputs.script
        results = self.mlab.run()
        runtime.returncode = results.runtime.returncode
        if self.mlab.inputs.uses_mcr:
            if 'Skipped' in results.runtime.stdout:
                self.raise_exception(runtime)
        runtime.stdout = results.runtime.stdout
        runtime.stderr = results.runtime.stderr
        #runtime.merged = results.runtime.merged
        return runtime

    def fix_mscript(self, mscript):
        """ugly hack to make work with current structue...
        need to turn data.scans data.pmscan into dict"""
        #newscript = mscript.replace('data.','data{1}.')
        newscript = mscript.replace('{1}.','.')
        newscript = newscript.replace('jobs.', 'jobs{1}.')
        newscript = newscript.replace('.spatial.', '.spm.spatial.')
        newscript = newscript.replace('\n{...\n', '\n')
        newscript = newscript.replace('\n};\n}','\n}')
        return newscript

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm
        """
        if opt == 'in_files':
            infiles =  scans_for_fnames(val,
                                        keep4d=True,
                                        separate_sessions=True)
            #return self._reformat_dict_for_savemat(dict(scans = infiles))
            return infiles
        #if opt == 'phase_map':
        #    infiles = val
        #    return self._reformat_dict_for_savemat(dict(pmscan = infiles))
        
        if opt == 'register_to_mean': # XX check if this is necessary
            return int(val)
        return val
    """
    def _parse_inputs(self):
        validate spm realign options if set to None ignore
                einputs = super(RealignUnwarp, self)._parse_inputs()
        #return [{'%s' % (self._jobtype):einputs[0]}]
        return einputs
    """
    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.in_files):
            outputs['realignment_parameters'] = []
        for imgf in self.inputs.in_files:
            if isinstance(imgf,list):
                tmp_imgf = imgf[0]
            else:
                tmp_imgf = imgf
            outputs['realignment_parameters'].append(fname_presuffix(tmp_imgf,
                                                                     prefix='rp_',
                                                                     suffix='.txt',
                                                                     use_ext=False))
            if not isinstance(imgf,list) and func_is_3d(imgf):
                break;
        #if self.inputs.jobtype == "write" or self.inputs.jobtype == "estwrite":
        if isinstance(self.inputs.in_files[0], list):
            first_image = self.inputs.in_files[0][0]
        else:
            first_image = self.inputs.in_files[0]

        outputs['mean_image'] = fname_presuffix(first_image, prefix='meanu')
        outputs['realigned_files'] = []
        # get prefix for new files, or default 'u'
        file_prefix = self.inputs.write_prefix or 'u'
        for imgf in filename_to_list(self.inputs.in_files):
            realigned_run = []
            if isinstance(imgf,list):
                for inner_imgf in filename_to_list(imgf):
                    realigned_run.append(fname_presuffix(inner_imgf, prefix=file_prefix))
            else:
                realigned_run = fname_presuffix(imgf, prefix=file_prefix)
                outputs['realigned_files'].append(realigned_run)
        return outputs


