PREPROCESSING GUIDE
===================

https://github.com/kbegany/jagust_rsfmri


Outline
-------

#. Realignment
#. Registration
#. Anatomical tissue segmentation
#. Slice-time correction
#. Bandpass filter
#. Regression model


Detailed Procedure
------------------


Despiking
+++++++++

**AFNI implementation:**

* rsfmri/examples/despike.py
* rsfmri/utils/afni_despike

Uses afni Despike to despike a 4d dataset, saves as ds_<filename>.


Realignment & Unwarping
+++++++++++++++++++++++

**SPM implementation:**

* rsfmri/examples/realignst_spm.py

Registers anatomical image to the mean functional image. Uses
spm_realign to estimate translation and rotation parameters using
within-modality rigid body alignment and unwarps by attempting to
remove movement-related variance from the timeseries.  Saves motion
parameters as an Excel file. SPM slice-time correction follows.

* Slice-time correction
* Despike option

http://www.fil.ion.ucl.ac.uk/spm/toolbox/unwarp/


**ANTS implementation:**
(i.e. Advanced Normalization Tools)

* rsfrmi/examples/realign_ants.py

Uses cross correlation with rigid transform to create an affine
mapping for registeration of functional images to the first functional
image. Reslices data to target space; default is linear interpolation
(nearest-neighbor option). Then collates the affine transform to
generate an Excel file with the motion parameters.

* No slice-time correction - Can be added using the nibabel implementation 
* Despike option

http://stnava.github.io/ANTs/



Anatomical tissue segmentation
++++++++++++++++++++++++++++++

**Freesurfer implementation:**

Segmentation of grey and white matter from anatomical image using to
generate brainmask (bmask) and anatomical parcellation (aparc) files.
The bmask aids in generating a mean functional image, which is used
during coregistration. The coregistered aparc is used to create masks
of the (kernel-eroded) white matter and ventricles during coregistration.


Coregistration
++++++++++++++

**ANTs implementation:**
(i.e. Advanced Normalization Tools)

* rsfmri/examples/coreg_anat_regressors.py

Uses mutual information to create an affine mapping between the mean
functional image and the brainmask.  This transform is inverted and
applied to coregister the brainmask and the anatomical parcellation to
the mean functional image.

**Optional SPM implementation:**

* rsfmri/utils/spm_coregister

Uses SPM coregister.


Slice-time correction
+++++++++++++++++++++

**SPM and Nibabel implementation:**

* rsfmri/examples/slicetime.py
* rsfmri/utils/get_slicetime
* rsfmri/utils/spm_slicetime

Utilizes number of slices, acquisition time (TA), repetition time
(TR), slice order (indicates if even or odd slices collected first) to
find the order of slice acqusition using Nibabel.  Performs slice-time
correction using SPM SliceTiming, with middle slice as reference.


Bandpass filter
+++++++++++++++


**FSL and Nitime implementation implementation:**

* rsfrmi/examples/bandpass.py
* rsfmri/utils/fsl_bandpass

Uses fslmaths to bandpass filter a 4d file.  Default lowf=0.0083, highf=0.15.

* rsfmri/utils/nitime_bandpass

Uses nitime to bandpass regressors.  Default ub=0.15, lb=0.0083.

* Regressors: Global signal option


Regression model
++++++++++++++++

**Freesurfer Implementation:**

* rsfmri/examples/model.py
* rsfmri/utils/run_feat_model
* rsfmri/utils/update_fsf
* rsfmri/utils/update_fsf
* rsfmri/utils/run_film

Uses FSL's feat_model, which uses the fsf file to generate files
necessary to run FILMGLS to fit design matrix to timeseries.

* Regressors: Global signal option
