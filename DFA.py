import numpy as np

try:
    import cvxopt
    from l1 import l1
    __cvxopt = True
except ImportError:
    __cvxopt = False


def sw_view(X, win, step):
    """
    Make a sliding window view of X with 50% overlap.

    X is a channels x samples array alternatively, X may be 1d as well.

    This is manipulating the strides of the original array for fast access.
    """
    # X must be C contiguous for the function to work
    if not X.flags['C_CONTIGUOUS']:
        X = X.copy()
    assert X.flags['C_CONTIGUOUS'], 'array must be c_contiguous'
    # get the total number of windows
    win_num = (X.shape[-1] - win) // step + 1
    if X.ndim == 1:
        XX = np.lib.stride_tricks.as_strided(
                X,
                shape=[win_num]+[win],
                strides=[X.dtype.itemsize * step] + [X.dtype.itemsize])
    elif X.ndim == 2:
        XX = np.lib.stride_tricks.as_strided(
                X,
                shape=[X.shape[0]]+[win_num]+[win],
                strides=([X.strides[0]] + [X.dtype.itemsize*step] +
                         [X.dtype.itemsize]))
    else:
        raise TypeError('X must be 1d or 2d numpy array')
    return XX


def detrend_l1(b, deg=1):
    """
    Return the detrended version of b along the last axis.

    This function minimizes the l1 norm and is thus more robust to outliers
    than OLS regression.

    deg - degree of the polynomial to remove
    """
    t = np.linspace(0, 1, b.shape[-1], endpoint=False, dtype=float)
    A = cvxopt.matrix(np.array([t**o_now for o_now in range(deg + 1)]).T)
    if b.ndim == 1:
        x = l1(A, cvxopt.matrix(b))
    else:
        x = np.hstack([np.array(l1(A, cvxopt.matrix(b_now)))
                       for b_now in b.reshape(-1, b.shape[-1])])
    fit = np.dot(A, x).T.reshape(b.shape)
    return b - fit


def detrend(X, deg=1):
    """
    Return the detrended version of Y along the last axis.

    deg - degree of the polynomial to remove
    """
    x = np.linspace(0, 1, X.shape[-1], endpoint=False, dtype=float)
    C = np.polynomial.polynomial.polyfit(
            x, X.T.reshape(X.shape[-1], -1), deg=deg)
    return (X - np.polynomial.polynomial.polyval(
        x, C).T.reshape(X.T.shape).T)


def DFA(x, windows, deg=1, l1=False):
    """
    Calculate the DFA.

    Args:
        - x (ndarray):  a channels x samples array, or alternatively a
                        1d array
        - windows (iterables): the windows in which the dfa should be
                               calculated
        - deg (int): the order of the polynomial that is removed.
                     Defaults to linear detrending (deg = 1)
        - l1 (bool): Whether the l1 loss should be used for calculating
                     the detrending. This should make the analysis more
                     robust to outliers at the expense of slower
                     calculations. If yes, cvxopt needs to be installed.
                     Defaults to False.

    Returns:
        - dfa (ndarray): the fluctuation value per detrending window.
                         shape will be channels x windows
    """
    # initialize output array
    if x.ndim == 1:
        dfa = np.empty([len(windows)], float)
    elif x.ndim == 2:
        dfa = np.empty([len(windows), x.shape[0]], float)
    else:
        raise TypeError('x must be 1d or 2d')
    if l1 not in [0, 1, True, False]:
        raise TypeError('l1 must be a boolean')
    l1 = bool(l1)
    if l1 and not __cvxopt:
        raise ImportError('cvxopt and the script l1.py are needed for l1' +
                          ' norm minimization')
    #############################
    # Start Calculation of DFA #
    # calculate the cumulative sum after removing the mean /
    if l1:
        x = np.cumsum(detrend_l1(x, deg=0), axis=-1)
    else:
        x = np.cumsum(detrend(x, deg=0), axis=-1)
    #############################
    # make sure that x is contiguous, so that
    # the array is not copied
    if not x.flags['C_CONTIGUOUS']:
        x = x.copy()
    for i, win_now in enumerate(windows):
        xx = sw_view(x, win_now, win_now // 2)
        if l1:
            dfa[i] = np.abs(detrend_l1(xx, deg=deg)).mean(-1).mean(-1)
        else:
            dfa[i] = np.sqrt((detrend(xx, deg=deg)**2).mean(-1).mean(-1))
    return dfa.T


def DFA_exponent(windows, dfa):
    """
    Calculate the DFA exponent for the supplied windows and
    the DFA fluctuation values.

    Args:
        windows (iterable): window lengths the dfa had been calculated for
        dfa (ndarray): fluctuation per window, if 2d, first dimension is
                       channels and second is fluctuations per window

    Returns:
        dfa_exponents (float or ndarray): the estimated exponents
        R2 (float or ndarray): the r squared value indicating the
                               goodness of fit
    """
    dfa = dfa.T
    C = np.polynomial.polynomial.polyfit(
            np.log(windows),
            np.log(dfa).reshape(dfa.shape[0], -1), deg=1)
    # get goodness of fit
    resids = np.log(dfa) - np.polynomial.polynomial.polyval(
            np.log(windows), C).T.reshape(dfa.shape)
    R2 = 1 - np.var(resids, axis=0) / np.var(np.log(dfa), axis=0)
    slopes = C[1]
    if dfa.ndim == 1:
        return float(slopes), R2
    else:
        return slopes.reshape(dfa.shape[1:]), R2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # generate noise, exponent should be 0.5
    size = 100000
    x = np.random.randn(size)
    # generate windows (linearly ascending in log-log space)
    windows = np.exp(np.linspace(np.log(5), np.log(10000), 20)).astype(int)
    dfa = DFA(x, windows)
    exp, R2 = DFA_exponent(windows, dfa)

    fig, ax = plt.subplots(2, 3, figsize=[20, 10])
    # plot data
    ax[0, 0].plot(x)
    ax[0, 0].set_title('Gausian Noise without outliers')
    ax[0, 0].set_xlabel('time')
    ax[0, 0].set_ylabel('amplitude')
    # plot histogram of x values
    ax[0, 1].hist(x, 500)
    ax[0, 1].set_xlabel('amplitude')
    ax[0, 1].set_xlabel('count')
    ax[0, 1].set_title('Histogram of data values')
    # plot dfa log-log-plot
    ax[0, 2].loglog(windows, dfa, 'bo-',
                 label='$exp={0:.2f}$, $R^2={1:.2f}$'.format(exp, R2))
#    ax[0, 2].axis('equal')
    ax[0, 2].set_title('DFA slope estimate')
    ax[0, 2].set_xlabel('window')
    ax[0, 2].set_ylabel('fluctuation per window')
    ax[0, 2].legend()

    import colorednoise as cn
    slope = 0.5
    x = cn.powerlaw_psd_gaussian(slope, size)
    # generate windows (linearly ascending in log-log space)
    windows = np.exp(np.linspace(np.log(5), np.log(10000), 20)).astype(int)
    dfa = DFA(x, windows)
    exp, R2 = DFA_exponent(windows, dfa)
    ax[1, 0].plot(x)
    ax[1, 0].set_title(f'Pink 1/{slope}-noise without outliers')
    ax[1, 0].set_xlabel('time')
    ax[1, 0].set_ylabel('amplitude')
    # plot histogram of x values
    ax[1, 1].hist(x, 500)
    ax[1, 1].set_xlabel('amplitude')
    ax[1, 1].set_xlabel('count')
    ax[1, 1].set_title('Histogram of data values')
    # plot dfa log-log-plot
    ax[1, 2].loglog(windows, dfa, 'bo-',
                 label='$exp={0:.2f}$, $R^2={1:.2f}$'.format(exp, R2))
#    ax[1, 2].axis('equal')
    ax[1, 2].set_title('DFA slope estimate')
    ax[1, 2].set_xlabel('window')
    ax[1, 2].set_ylabel('fluctuation per window')
    ax[1, 2].legend()
    fig.tight_layout()
    plt.show()
