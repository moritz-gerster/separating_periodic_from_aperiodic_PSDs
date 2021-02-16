import numpy as np
import scipy.signal
import pdb

try:
    from tqdm import trange
except:
    trange = range


def stepdown_p(stat, stat_boot):
    """
    Calculate FWER corrected p values in multiple comparison problems.

    This implements the method from:

    Romano, J.P., and Wolf, M. (2016). Efficient computation of adjusted
    p-values for resampling-based stepdown multiple testing.
    Stat. Probab. Lett. 113, 38–40.

    Input:
        stat - array shape N - "true" values of the test statistic
        stat_boot - array shape M x N - the values of the test statistic
            obtained from M bootstraps (or permutations, or ...)
    Returns:
        p_values
    """
    M, N = stat_boot.shape
    if not N == len(stat):
        raise ValueError('length of stat must match number of variables'
                ' in stat_boot')
    # order the test hypotheses with decreasing significance 
    order = np.argsort(stat)[::-1]
    stat = stat[order]
    stat_boot = stat_boot[:,order]
    # initialize results array
    p = [(np.sum(np.max(stat_boot[:,i:], 1) >= stat[i]) + 1)/float(M + 1)
            for i in trange(N)]
    # enforce monotonicity
    p = np.maximum.accumulate(p)
    # revert the original order of the hypothesis
    return p[np.argsort(order)]

def bootstrap_coherence_diff(x, y, nperseg, mag=True, imag=True,
        N_bootstrap=1000, fs=1.0, axis=-1, **kwargs):
    """Calculate a bootstrap t statistic for the difference in coherence
    between dataset x and y
    
    Welch's average periodogram technique is used
    
    Args:
        x (ndarray): a 2d numpy array whose csd is to be calculated
        y (ndarray): a 2d numpy array whose csd is to be calculated. The shape
                     of x and y must be equal except for the axis indicated by
                     the axis argument
        nperseg (int): the number of datapoints in every segment,
            determines the frequency resolution
        mag (bool): whether bootstrap should be calculated for the magnitude
            squared coherence
        imag (bool): whether bootstrap should be calculated for the imaginary
            part of coherence
        N_bootstrap (int): number of bootstrap resamples, defaults to 1000
        fs (float): the sampling rate
        axis (int): the axis along which the Fourier transform is
            calculated. Defaults to the last axis
        **kwargs: Keyword arguments, as in scipy.signal.welch

    Returns:
        f (ndarray): frequencies corresponding to the result array
        coherence (ndarray) coherence values (real and imaginary part)
        (t_mag, t_bootstrap_mag): (tuple of ndarray): only if mag is True
            t value for the differences in magnitude coherence for original
            dataset and for the bootstrap resamples
        (t_imag, t_bootstrap_imag): (tuple of ndarray): only if imag is True
            t value for the differences in imaginary part of coherence for
            original dataset and for the bootstrap resamples
    """
    if x.ndim != 2:
        raise ValueError('x must be a 2d ndarray')
    if y.ndim != 2:
        raise ValueError('y must be a 2d ndarray')
    if axis != -1:
        x = np.rollaxis(x, axis, len(x.shape))
        y = np.rollaxis(y, axis, len(y.shape))
    if not x.shape[0] == y.shape[0]:
        raise ValueError('shape of x and y must be equal except for the' +
                         ' axis indicated by the axis argument')
    idx1, idx2 = np.indices([x.shape[0], x.shape[0]])
    f, x_t, x_csd = scipy.signal.spectral._spectral_helper(x[idx1], x[idx2],
            fs=fs, nperseg=nperseg, axis=-1, mode='psd', **kwargs)
    f, y_t, y_csd = scipy.signal.spectral._spectral_helper(y[idx1], y[idx2],
            fs=fs, nperseg=nperseg, axis=-1, mode='psd', **kwargs)
    # reject temporal segments with NaNs of infs
    x_csd = x_csd[...,np.all(np.isfinite(x_csd),
        axis=tuple(range(x_csd.ndim - 1)))]
    y_csd = y_csd[...,np.all(np.isfinite(y_csd),
        axis=tuple(range(y_csd.ndim - 1)))]
    if (mag and imag):
        t_mag, t_imag  = _jackknife_coherence_diff(x_csd, y_csd, mag, imag,
                z_transform=True)
        t_mag_boot = []
        t_imag_boot = []
    else:
        t  = _jackknife_coherence_diff(x_csd, y_csd, mag, imag,
                z_transform=True)
        t_boot = []
    # calculate bootstrap
    N_x = x_csd.shape[-1]
    N_y = y_csd.shape[-1]
    N = N_x + N_y
    all_csd = np.concatenate([x_csd, y_csd], axis=-1)
    for _ in trange(N_bootstrap):
        idx1 = np.random.randint(0, N, N_x)
        idx2 = np.random.randint(0, N, N_y)
        if (mag and imag):
            t_mag_temp, t_imag_temp = _jackknife_coherence_diff(
                    all_csd[...,idx1],
                    all_csd[...,idx2],
                    mag, imag, z_transform=True)
            t_mag_boot.append(t_mag_temp)
            t_imag_boot.append(t_imag_temp)
        else:
            t_boot.append(_jackknife_coherence_diff(
                    all_csd[...,idx1],
                    all_csd[...,idx2],
                    mag, imag, z_transform=True))
    if (mag and imag):
        return (t_mag, np.array(t_mag_boot)), (t_imag, np.array(t_imag_boot))
    else:
        return (t, t_boot)

def _jackknife_coherence_diff(x_csd, y_csd, mag=True, imag=True,
        z_transform=True):
    '''
    Internal function, to not use directly
    Calculates a t value for the difference between the coherence between
    datasets x and y

    Args:
    x_csd (ndarray): shape channel x channel x frequencies x temporal segments
    y_csd (ndarray): shape channel x channel x frequencies x temporal segments
    mag (bool): whether magnitude coherence should be calculated
                Defaults to True
    imag (bool): whether imaginary part of coherence should be calculated
                 Defaults to True
    z_transform (bool): Whether Fisher's z-transform should be calculanted for
                        the estimates of coherence. This seems to be preferable.
                        Defaults to True
    '''
    # Calculate the jackknife estimates of the cross-spectral densities of x
    # and y
    if not (mag or imag):
        raise ValueError('any or both of mag or imag must be True')
    x_csd_jackknife = (x_csd.sum(-1)[...,np.newaxis] - x_csd)/(
            x_csd.shape[-1] - 1)
    y_csd_jackknife = (y_csd.sum(-1)[...,np.newaxis] - y_csd)/(
            y_csd.shape[-1] - 1)
    x_coherence = calc_coherence(x_csd_jackknife)
    y_coherence = calc_coherence(y_csd_jackknife)
    if mag:
        x_mag = np.abs(x_coherence)
        y_mag = np.abs(y_coherence)
        if z_transform:
            x_mag = np.arctanh(x_mag)
            y_mag = np.arctanh(y_mag)
        x_mag_mean = x_mag.mean(-1)
        x_mag_var = (x_csd.shape[-1] - 1) / x_csd.shape[-1] * np.sum(
                (x_mag - x_mag_mean[...,np.newaxis])**2, -1)
        y_mag_mean = y_mag.mean(-1)
        y_mag_var = (y_csd.shape[-1] - 1) / y_csd.shape[-1] * np.sum(
                (y_mag - y_mag_mean[...,np.newaxis])**2, -1)
        # calculate a t statistic with the formula for a welch test
        t_mag = ((x_mag_mean - y_mag_mean) / 
                np.sqrt(x_mag_var / x_csd.shape[-1] +
                        y_mag_var / y_csd.shape[-1]))
    if imag:
        x_imag = np.imag(x_coherence)
        y_imag = np.imag(y_coherence)
        if z_transform:
            x_imag = np.arctanh(x_imag)
            y_imag = np.arctanh(y_imag)
        x_imag_mean = x_imag.mean(-1)
        x_imag_var = (x_csd.shape[-1] - 1) / x_csd.shape[-1] * np.sum(
                (x_imag - x_imag_mean[...,np.newaxis])**2, -1)
        y_imag_mean = y_imag.mean(-1)
        y_imag_var = (y_csd.shape[-1] - 1) / y_csd.shape[-1] * np.sum(
                (y_imag - y_imag_mean[...,np.newaxis])**2, -1)
        # calculate a t statistic with the formula for a welch test
        t_imag = ((x_imag_mean - y_imag_mean) / 
                np.sqrt(x_imag_var / x_csd.shape[-1] +
                        y_imag_var / y_csd.shape[-1]))
    if (mag and imag):
        return t_mag, t_imag
    elif mag:
        return t_mag
    else:
        return t_imag

def calc_csd(x, nperseg, fs=1.0, axis=-1, average='mean', jackknife=False,
        **kwargs):
    """Calculate the cross-spectral density (csd) of a 2d signal
    Welch's average periodogram technique is used

    Args:
        x (ndarray): a 2d numpy array whose csd is to be calculated
        nperseg (int): the number of datapoints in every segment,
            determines the frequency resolution
        fs (float): the sampling rate
        axis (int): the axis along which the Fourier transform is
            calculated. Defaults to the last axis
        average (str or func): the type of average calculated across
            segments. Can be 'mean' or 'median' or a function, accepting
            the Fourier-transformed data segments as input
        jackknife (bool): Whether a jackknife estimate of the variance of
            the csd estimation (and coherence) should be returned.
            If true, average must be 'mean'
        **kwargs: Keyword arguments, as in scipy.signal.welch

    Returns:
        f (ndarray): frequencies corresponding to the result array
        csd (ndarray): cross-spectral density of x. The shape will be
            (x.shape[-1 - axis], x.shape[-1 - axis], len(f))
            if jackknife is True, this is a tuple additionally returnining
            an estimate of the variance
    """
    if jackknife:
        if not average == 'mean':
            raise ValueError('If jackknifing is to be performed, ' + 
                    'average must be \'mean\'')
    if average == 'mean':
        def average(x):
            return np.nanmean(x, axis=-1)
    if average == 'median':
        def average(x):
           return (np.median(x, axis=-1) /
                   scipy.signal.spectral._median_bias(x.shape[-1]))
    if x.ndim != 2:
        raise ValueError('x must be a 2d ndarray')
    if axis != -1:
        x = np.rollaxis(x, axis, len(x.shape))
    idx1, idx2 = np.indices([x.shape[0], x.shape[0]])
    f, t, csd = scipy.signal.spectral._spectral_helper(x[idx1], x[idx2],
            fs=fs, nperseg=nperseg, axis=-1, mode='psd', **kwargs)
    if jackknife:
        # in this case, average is always 'mean'
        csd_jackknife = (csd.sum(-1)[...,np.newaxis] - csd)/(
                csd.shape[-1] - 1)
        csd_mean = csd_jackknife.mean(-1)
        csd_var = (len(t) - 1) / len(t) * np.sum(
                (csd_jackknife - csd_mean[...,np.newaxis])**2, -1)
        return f, (csd_mean, csd_var)
    else:
        # calculate the requested average
        try:
            csd_mean = average(csd)
        except(TypeError):
            'average must be a function, got %' % type(average)
        else:
            return f, csd_mean

def trimmed_mean_cov(x, proportiontocut = (0, 0.1)):
    '''Calculate the the average of a number of covariance matrices after
    trimming the distribution of total power at the left and right tails.

    Is calculated along the last axis of the input data

    Args:
        x (ndarray): an array of covariance matrices with shape
            nchans x nchans x ... where nchans is the number of channels
        proportionticut (tuple): a 2-tuple with the (0,1) proportion of the
            distribution at the lower/upper end of the distribution.
            The number of removed samples is round to the next lower integer
    '''
    # after spectral_helper, x has the shape
    # nchans x nchans x nfreqs x nsegments
    # -> we want to calculate the total power across the diagonal
    # of the nchans x nchans segments
    if not x.shape[0] == x.shape[1]:
        raise ValueError('x must have equal sizes of first and second axes ' +
                'got %s' % x.shape[:2])
    if not x.ndim > 2:
        raise ValueError('x must have more than 2 dimensions')
    nchans = x.shape[0]
    nsamples = x.shape[-1]
    start = int(proportiontocut[0]*nsamples)
    stop = nsamples - int(proportiontocut[1]*nsamples)
    # calculate the power of every segment and get indices of the segments
    # with power sorted in ascending order
    #calculate the total power of every segment (without the dc-component)
    power = x[range(nchans), range(nchans), 1:].sum(0).sum(0).real
    idx = np.argsort(power, axis=-1)
    chosen_idx = idx[...,start:stop]
    # calculate the mean after trimming
    # in the current version to calculate the trim across the total power
    # of every segment with a 1d idx-variable, the following line is overly
    # complex, however, works well even for 2d idx variables...
    sorted_x = np.take_along_axis(x,
            chosen_idx[np.newaxis, np.newaxis, np.newaxis], axis=-1)
    return sorted_x.mean(-1)

def calc_coherence(csd):
    """Calculate the coherence from a csd-mactrix

    Args:
        csd (complex ndarray): matrix of cross-spectral density matrices

    Returns:
        coherence (ndarray): complex valued coherence matrix
    """
    if not csd.shape[0] == csd.shape[1]:
        raise ValueError('csd must be square in the first 2 dimensions')
    ch = csd.shape[0]
    return csd/np.sqrt(csd[range(ch), range(ch),np.newaxis]*
            csd[np.newaxis, range(ch), range(ch)])

