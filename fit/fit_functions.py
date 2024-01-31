import numpy as np

from lmfit.models import ExponentialGaussianModel, ConstantModel, GaussianModel

def Gauss(params, i, x):
    """calc gaussian from params for data set i
    using simple, hardwired naming convention"""
    model = GaussianModel(prefix='f_%i_' % (i)) + ConstantModel(prefix='f_%i_' % (i))
    return model.eval(params, x=x)

def DoubleGauss(params, i, x):
    """calc gaussian from params for data set i
    using simple, hardwired naming convention"""
    model = GaussianModel(prefix='f_%i_' % (i)) + GaussianModel(prefix='f_%i_dg_' % (i)) + ConstantModel(prefix='f_%i_' % (i))
    return model.eval(params, x=x)

def TripleGauss(x, a, x0, sigma0, b,x1, sigma1, c, x2, sigma2, bias):
    return bias+a*np.exp(-(x-x0)**2/(2*sigma0**2))+b*np.exp(-(x-x1)**2/(2*sigma1**2))+c*np.exp(-(x-x2)**2/(2*sigma2**2))

def Convol(x, a, x0, sigma,bias, tau, x1, b):
    return bias + convolve_custom((a/sigma)*np.exp(-(x-x0)**2/(2*sigma**2)), (b/tau)*np.exp(-(x-x0)/tau))

#def DoublegaussConvol(x, a, x0, sigma0, b, x1, sigma1, tau, x2, c, bias):
#    return bias + (c/(tau*sigma0))*convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x2)/tau)) + (b/sigma1)*np.exp(-(x-x1)**2/(2*sigma1**2))

def DoublegaussConvol(x, a, x0, sigma0, b, sigma1, x1, tau, x2, bias):
    return bias + (a/(tau*sigma0))*convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x2)/tau)) + (b/sigma1)*np.exp(-(x-x1)**2/(2*sigma1**2))


def DoublegaussConvolBackground(x, a, x0, sigma0, sigma1, tau0, tau1, bias):
    return bias + (a/(sigma0*sigma1*tau0*tau1))*convolve_custom(convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x0)/tau0)) + np.exp(-(x-x0)**2/(2*sigma1**2)), np.exp(-(x-x0)/tau1))

def convolve_custom(arr, kernel):
    """Simple convolution of two arrays."""
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff+npts]

