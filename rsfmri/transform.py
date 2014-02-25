import os, math
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import pandas

import nibabel as ni

_EPS = np.finfo(float).eps * 4.0

def ants_to_matrix(infile):
    """ takes a transform defined by ANTS and turns it into a 
    4X4 affine transform matrix"""
    for line in open(infile):
        if 'FixedParameters' in line:
            fixed_parts = line.replace('\n','').split(' ')
            continue
        if 'Parameters:' in line:
            parts = line.replace('\n','').split(' ')
    rot = np.asarray([float(x) for x in parts[1:10]])
    rot.shape = (3,3)
    trans = np.asarray([float(x) for x in parts[-3:]])
    
    affine = np.eye(4)
    affine[:3,:3] = rot
    affine[:-1,-1] = trans
    return affine, np.array(fixed_parts[1:], dtype=float)


def vector_lengths( X, P=2., axis=None ):
    """
    Finds the length of a set of vectors in *n* dimensions.  
    This is like the :func:`numpy.norm` function for vectors, 
    but has the ability to work over a particular axis of 
    the supplied array or matrix.
    Computes ``(sum((x_i)^P))^(1/P)`` for each ``{x_i}`` 
    being the elements of *X* along the given axis.  
    If *axis* is *None*,compute over all elements of *X*.
    """
    X = np.asarray(X)
    return (np.sum(X**(P),axis=axis))**(1./P)


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix([1, 2, 3])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3, ))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective

def params_from_ants(infile):
    aff, center = ants_to_matrix(infile)
    center4 = np.ones(4)
    center4[:-1] = center
    scale, shear, angles, translate, perspective = decompose_matrix(aff)

    displacement = vector_norm(center4 - np.dot(aff, center4))
    return displacement, translate, angles


def d4_to_d2(data4d_file):
    """ take 4d data (x,y,x,t) and return a 
    (t, nvox) shaped data array"""
    img = ni.load(data4d_file)
    dat = img.get_data().squeeze() # also remove singular dims
    orig_shape = dat.shape
    if not len(orig_shape) == 4:
        raise ValueError('expected 4D data not %s-D'%(orig_shape))
    newshape = (orig_shape[-1], np.prod(orig_shape[:-1]))
    # trans pose so time is first dim
    # deals with non-contiguous data issue
    dat = dat.T
    dat.shape = newshape
    return dat, orig_shape
   
def test_d4_to_d2():
    """ test function """
    tmpdir = tempfile.mkdtemp()
    orig_shape = (10,11,12,20)
    testdat = np.zeros(orig_shape)
    img = ni.Nifti1Image(testdat, np.eye(4))
    tmpfile = os.path.join(tmpdir, 'testdat.nii.gz')
    img.to_filename(tmpfile)
    expected = (20, 10*11*12)
    changed, inshape = d4_to_d2(tmpfile)
    assert inshape == orig_shape
    assert changed.shape == expected
    os.unlink(tmpfile)


def detrend_data(data4d_file):
    """takes 4d data (x, y, z, t) and detrends
    combines the first dims (x,y,z) and returns a
    (nvoxels, timepoints) dimension image"""
    dat, orig_shape = d4_to_d2(data4d_file)
    detrended = ss.detrend(dat, axis=1)
    return detrended, orig_shape

def calc_snr(data4d):
    """ takes in array (timepoints by nvoxels)
    calcs snr across time, and returns resulting
    snr array"""
    pth, nme = os.path.split(data4d)
    img = ni.load(data4d)
    aff = img.get_affine()
    detrended, orig_shape = detrend_data(data4d)
    snr = detrended.mean(axis=0) / detrended.std(axis=0) 
    snr.shape  = orig_shape[:-1][::-1]
    snrimg = ni.Nifti1Image(snr.T, aff)
    snrfile = os.path.join(pth, 'snr.nii.gz')
    snrimg.to_filename(snrfile)
    return snrfile

def decompose(M):
    """Decompose homogenous affine transformation matrix into parts.
    The parts are translations, rotations, zooms, shears.
    M can be any square matrix, but is typically shape (4,4)
    Decomposes M into ``T, R, Z, S``, such that, if M is shape (4,4)::

       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A = np.eye(4)
       A[:3,:3] = RZS
       A[:-1,-1] = T

    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.

    The case above (A.shape == (4,4)) is the most common, and
    corresponds to a 3D affine, but in fact A need only be square.

    Parameters
    ----------
    M : array shape (N,N)
    
    Returns
    -------
    T : array, shape (N-1,)
       Translation vector
    R : array shape (N-1,N-1)
        rotation matrix
    Z : array, shape (N-1,)
       Zoom vector.  May have one negative zoom to prevent neeed for
       negative determinant R matrix above
    S : array, shape (P,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine.
       
    Examples
    -------- 
    >>> T = [20, 30, 40] # translations
    >>> R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
    >>> Z = [2.0, 3.0, 4.0] # zooms
    >>> S = [0.2, 0.1, 0.3] # shears
    >>> # Now we make an affine matrix
    >>> A = np.eye(4)
    >>> Smat = np.array([[1, S[0], S[1]],
    ...                  [0,    1, S[2]],
    ...                  [0,    0,    1]])
    >>> RZS = np.dot(R, np.dot(np.diag(Z), Smat))
    >>> A[:3,:3] = RZS
    >>> A[:-1,-1] = T # set translations
    >>> Tdash, Rdash, Zdash, Sdash = decompose(A)
    >>> np.allclose(T, Tdash)
    True
    >>> np.allclose(R, Rdash)
    True
    >>> np.allclose(Z, Zdash)
    True
    >>> np.allclose(S, Sdash)
    True

    Notes
    -----
    We have used a nice trick from SPM to get the shears.  Let us call
    the starting N-1 by N-1 matrix ``RZS``, because it is the
    composition of the rotations on the zooms on the shears.  The
    rotation matrix ``R`` must have the property ``np.dot(R.T, R) ==
    np.eye(N-1)``.  Thus ``np.dot(RZS.T, RZS)`` will, by the transpose
    rules, be equal to ``np.dot((ZS).T, (ZS))``.  Because we are doing
    shears with the upper right part of the matrix, that means that
    the cholesky decomposition of ``np.dot(RZS.T, RZS)`` will give us
    our ``ZS`` matrix, from which we take the zooms from the diagonal,
    and the shear values from the off-diagonal elements.
    """
    M = np.asarray(M)
    T = M[:-1,-1]
    RZS = M[:-1,:-1]
    ZS = np.linalg.cholesky(np.dot(RZS.T,RZS)).T
    Z = np.diag(ZS)
    shears = ZS / Z[:,np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n,n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
        return T, R, Z, S



def aff2axangle(aff):
    """Return axis, angle and point from affine

    Parameters
    ----------
    aff : array-like shape (4,4)

    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation
    point : array shape (3,)
       point around which rotation is performed

    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> point = np.random.random(3) - 0.5
    >>> R0 = axangle2aff(direc, angle, point)
    >>> direc, angle, point = aff2axangle(R0)
    >>> R1 = axangle2aff(direc, angle, point)
    >>> np.allclose(R0, R1)
    True

    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    """
    R = np.asarray(aff, dtype=np.float)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return direction, angle, point



def collate_affines(filelist):
    """ for a set of Affine files, collates params into a list of 
    translation, rotation, and displacement values
    return array"""
    allmove = []
    for tmpf in filelist:
        displacement, translate, angles = params_from_ants(tmpf)
        allmove.append([translate.tolist() + angles + [displacement]])
    return np.asarray(allmove).squeeze()


def movementarr_to_pandas(move_array, outfile_xls=None):
    """" given an array of movment params, creates a pandas
    dataframe object
    will write to outfile_xls if it is passed
    """
    dimx, dimy = move_array.shape
    dims = ('x','y','z')
    cols = ['trans_%s'%x for x in dims] + ['rot_%s'%x for x in dims] 
    if dimy > 6:
        # array includes displacement
        cols += ['displacement'] 
    movedf = pandas.DataFrame(move_array, columns=cols)
    if not outfile_xls is None:
        movedf.to_excel(outfile_xls)
    return movedf


def plot_movement(movement, outdir):
    """ movment is an array (ntimepoints X 7)
    columns xtans, ytrans, ztrans, xrot, yrot, zrot, displacement
    (see collate_affines)
    """
    if not movement.shape[1] == 7:
        raise IOError('expected an nX7 matrix, found {0}'.format(movement.shape))
    # create subplots
    for val, data in enumerate((movement[:,:3], 
        movement[:,3:6], 
        movement[:,-1])):
        plt.subplot(3,1,val+1)
        if val >1:
            plt.plot(data, label = 'displacement')
            continue
        for item, axis in zip(data.T, ('x', 'y', 'z')):
            plt.plot(item, label = axis)
        plt.legend(loc = 'upper left')
    plt.savefig(os.path.join(outdir,'movement.png'))
    plt.clf()

