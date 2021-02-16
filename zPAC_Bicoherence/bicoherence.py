"""
This module calculates the bicoherence from the Fourier transform of
signals x, y, z

How bicoherence relates to phase-amlitude coupling is explained very well
in
@article{zandvoort2020understanding,
  title={Understanding phase-amplitude coupling from bispectral analysis},
  author={Zandvoort, Coen S and Nolte, Guido},
  journal={bioRxiv},:w
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}

"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

try:
    from numba import njit
    numba_exists = True

    @njit
    def _calcB(X, Y, Z, XisY, B):
        avg_over_t = (B.ndim == X.ndim)
        # loop through all elements
        for idx in np.ndindex(B.shape):
            if idx[0] + idx[1] >= len(B):
                pass
            elif XisY and idx[1] < idx[0]:
                pass
            else:
                if avg_over_t:
                    for t in range(X.shape[-1]):
                        B[idx] += (
                                X[(idx[0],) + idx[2:] + (t,)] *
                                Y[idx[1:] + (t, )] *
                                Z[(idx[0] + idx[1],) + idx[2:] + (t,)
                                    ].conjugate()) / X.shape[-1]
                else:
                    B[idx] = (
                            X[(idx[0],)+idx[2:]]*
                            Y[idx[1:]]*
                            Z[(idx[0] + idx[1],) + idx[2:]].conjugate()
                            )
        return B

    def _calcN(X, Y, Z, XisY, N):
        if N.ndim == X.ndim:
            return __calcN(
                    (np.abs(X)**3).mean(-1),
                    (np.abs(Y)**3).mean(-1),
                    (np.abs(Z)**3).mean(-1),
                    XisY, N)**(1/3.)
        else:
            return __calcN(
                    np.abs(X)**3,
                    np.abs(Y)**3,
                    np.abs(Z)**3,
                    XisY, N)**(1/3.)

    @njit
    def __calcN(X, Y, Z, XisY, N):
        # loop through all elements
        for idx in np.ndindex(N.shape):
            if idx[0] + idx[1] >= len(N):
                pass
            elif XisY and idx[1]<idx[0]:
                pass
            else:
                N[idx] = (
                        X[(idx[0],)+idx[2:]]*
                        Y[idx[1:]]*
                        Z[(idx[0] + idx[1],) + idx[2:]]
                        )
        return N

except ModuleNotFoundError:
    numba_exists = False

def bispectrum(X, Y, Z, f_axis=0, t_axis=-1, return_norm=True):
    """ Calculates cross-bispectrum of Fourier transforms X, Y, Z, obtaine 
    from signals x, y, z

    The bispectrum is than a matrix with indices (f1, f2), such that
    B(f1, f2) is <X(f1) * Y(f2) * conj(Z)> where <> is the expectation (e.g.
    across segments of the data) and conj is the conjugate transpose.

    It is supposed that the the frequencies in X, Y, Z are identical and
    in a regulary ascending order such that, if the frequency index of f1
    is idx1 and the frequency index of f2, the frequency index of (f1+f2) is
    (idx1 + idx2).

    Additionaly, the function can return a normalization which, if used as
    divisor, serves to calculate the the bicoherence.
    The normalization is calculated as:
    (<|X(f1)|**3> * <|Y(f2)|**3> * <|Z(f1 + f2)|**3>)**(1/3)

    The bicoherence effective measures the coupling between the phase of X(f1)
    and the amplitude of Y(f2) with f1<f2

    Args:
        X (complex ndarray): Fourier transform of x,
        shaped (...,f,..., t, ...)
        Y (complex ndarray): Fourier transform of y with same shape as X
        Z (complex ndarray): Fourier transform of z with same shape as X
        f_axis (int): the axis of X, Y, Z denoting the frequency axis
            (axis of f)
        t_axis (int or False): the axis of X, Y, Z denoting the
            axis that the bicoherence is averaged across (axis of t). If
            False, the data is understood to be from a single segment,
        return_norm: return a normalization for the bicoherence (see above)

    Returns:
        B (complex ndarray): cross-bispectrum between X, Y, Z for all
            possible combinations of f1 and f2, shaped (..., f1, f2, ...)
        N (real ndarray): normalization to calculate bicoherence as B/N, same
        shape as B, only returned if return_norm=True
    """
    XisY = X is Y
    if not X.shape == Y.shape == Z.shape:
        raise ValueError('the shapes of X, Y, and Z must be equal')
    if f_axis < 0:
        f_axis = X.ndim + f_axis
    if t_axis is not False:
        if t_axis < 0:
            t_axis = X.ndim + t_axis
        if t_axis == f_axis:
            raise ValueError(
                    'f_axis and t_axis must have different values')
        if t_axis < f_axis:
            t_axis += 1
    # move f_axis to axis 0 and t_axis to axis -1
    X = np.moveaxis(X, f_axis, 0)
    Y = np.moveaxis(Y, f_axis, 0)
    Z = np.moveaxis(Z, f_axis, 0)
    if t_axis is not False:
        X = np.moveaxis(X, t_axis, -1)
        Y = np.moveaxis(Y, t_axis, -1)
        Z = np.moveaxis(Z, t_axis, -1)
    f = X.shape[0]
    # create empty result arrays
    if t_axis is not False:
        if X.ndim > 2:
            B = np.zeros(np.r_[f, f, X.shape[1:-1]], np.complex)
        else:
            B = np.zeros([f,f], np.complex)
    else:
        B = np.zeros(np.r_[f, f, X.shape[1:]], np.complex)
    if return_norm:
        N = np.zeros(B.shape, float)
    if numba_exists:
        ##################################################
        # calculate the results in a fast numba function #
        ##################################################
        B = _calcB(X, Y, Z, XisY, B)
        if return_norm:
            N = _calcN(X, Y, Z, XisY, N)
    else:
        ###########################################################
        # calculate results using native numpy code using strides #
        ###########################################################
        # Note: strides are a pretty cool numpy feature!
        Z2 = as_strided(np.conj(Z),
                shape = (Z.shape[0],) + Z.shape,
                strides = (Z.strides[0],) + Z.strides,
                writeable=False)
        if t_axis is not False:
            B = 1/X.shape[t_axis] * np.einsum(
                    'i...t, j...t, ij...t -> ij...', X, Y, Z2)
            if return_norm:
                N = (
                        (np.abs(X)**3).mean(-1)[:,np.newaxis] *
                        (np.abs(Y)**3).mean(-1)[np.newaxis]*
                        (np.abs(Z2)**3).mean(-1)
                        )**(1/3)
        else:
            B = np.einsum('i..., j..., ij...-> ij...', X, Y, np.conj(Z2))
            if return_norm:
                N = (
                        (np.abs(X)**3)[:,np.newaxis] *
                        (np.abs(Y)**3)[np.newaxis]*
                        (np.abs(Z2)**3)
                        )**(1/3)
        # now, the right lower triangle of B and N is invalid (since
        # f1 + f2 >f),  and we need to set it to 0
        B = (B.T * np.tril(np.ones(2*(f,), float))[::-1]).T
        if return_norm:
            N = (N.T * np.tril(np.ones(2*(f,), float))[::-1]).T
    # reshape B and N
    if t_axis is not False:
        if t_axis <= f_axis:
            f_axis -= 1  # needed for finally shaping back
    B = np.moveaxis(B, (0, 1), (f_axis, f_axis + 1))
    if return_norm:
        N = np.moveaxis(N, (0, 1), (f_axis, f_axis + 1))
        return B, N
    return B
