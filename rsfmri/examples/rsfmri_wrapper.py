
from rsfmri import utils
from rsfmri import register

""" This is done in native space, add warped after (Renaud others)??  

split raw func
realign (no slicetime (ANTS))
realign w/slicetime (spm)
generate movement regressors
make meanfunc
files -> 4dfunc
(bias correct anat and meanfunc?)
register anat to meanfunc
pull whole brain, white, ventricle  rois (aparc)
erode white and ventricle
extract global, white, ventricle
bandpass filter 4ddata
bandpass filter regressors
generate fsf 
run model
grab residuals

"""









if __name__ == '__main__':
    print 'sample rsfmri'