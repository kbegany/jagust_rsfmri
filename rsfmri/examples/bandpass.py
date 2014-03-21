import pandas

from rsfmri import utils
from rsfmri import register

def find_make_workdir(subdir, despike, spm, logger=None):
    """ generates realign directory to query based on flags
    and if it exists and create new workdir
    """
    rlgn_dir = utils.defaults['realign_ants']
    if spm:
        rlgn_dir = utils.defaults['realign_spm']
    if despike:
        rlgn_dir = utils.defaults['despike'] + rlgn_dir
    rlgn_dir = os.path.join(subdir, rlgn_dir)
    if not os.path.isdir(rlgn_dir):
        if logger:
            logger.error('{0} doesnt exist skipping'.format(rlgn_dir))
        raise IOError('{0} doesnt exist skipping'.format(rlgn_dir))
    if logger:
        logger.info(rlgn_dir)
    workdirnme = utils.defaults['coreg']
    workdir, exists = utils.make_dir(rlgn_dir, workdirnme)
    if not exists:
        logger.error('{0}: skipping {1} doesnt exist'.format(subdir, workdir))
        raise IOError('{0}: MISSING, Skipping'.format(workdir))
    bpdirnme = utils.defaults['bandpass']
    bpdir, exists = utils.make_dir(workdir, bpdirnme)
    if exists:
        logger.error('{0}: skipping {1}  existS'.format(subdir, bpdir))
        raise IOError('{0}: EXISTS, Skipping'.format(bpdir))
    return rlgn_dir, workdir

def setup_logging(workdir, sid):
    logger = logging.getLogger('bandpass')
    logger.setLevel(logging.DEBUG)
    ts = reg.timestr()
    fname = os.path.split(__file__)[-1].replace('.py', '')
    logfile = os.path.join(workdir,
                           '{0}_{1}_{2}.log'.format(sid, fname, ts))
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(__file__)
    logger.info(ts)
    logger.info(workdir)
    logger.info(os.getenv('USER'))
    return logger


def get_file(workdir, type_glob, logger=None):
    files, nfiles = utils.get_files(workdir, type_glob)
    ## type_glob eg 'align4d_{0}.nii.gz'.format(sid) from utils.defaults)
    if logger:
        logger.info(' glob ({0}) yields nfiles: {1}'.format(type_glob, nfiles))
    if nfiles == 1:
        return files[0]
    logger.error('{0} in {1} not found'.format(type_glob, workdir))
    raise IOError('{0} in {1} not found'.format(type_glob, workdir))


def get_4d_func():
    pass


def get_regressors(rlgndir):
    xls = get_file(rlgndir, "B*movement.xls")
    df = pandas.ExcelFile(xls).parse('sheet1')
    ## ANTS movement missing padded 0's
    if 'ants' in rlgndir:
        df = utils.zero_pad_movement(df)
    # data is ntimepoints X nregressors
    # transpose and drop 7th item
    return df.values[:,:-1].T


def bandpass_func():
    pass

def bandpass_regressors():
    pass


def process_subject(subdir, despike=False, spm=False):

    _, sid = os.path.split(subdir)
    rawdir = os.path.join(subdir, 'raw')
    rlgndir, bpdir = find_make_workdir(subdir, despike, spm)
    logger = setup_logging(bpdir, sid)
    globtype = utils.defaults['aligned'].format(sid)
    aligned = get_file(rlgndir, globtype, logger)