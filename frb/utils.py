import numpy as np
import os
import fnmatch
from sklearn.mixture import DPGMM
from astropy.stats import mad_std, biweight_location

vround = np.vectorize(round)
vint = np.vectorize(int)


# TODO: Clipping out pixels with high `noise` flux can remove real FRB if it
# has high enough flux
def find_noisy(dsp, n_max_components, frac=100):
    """
    Attempt to characterize the noise of dynamical spectra. Fitting Dirichlet
    Process Gaussian Mixture Model to histogram of ``dsp`` values. If 2 (or
    more) components is chosen then we can find typical threshold for noise
    values.

    :param dsp:
        2D numpy.ndarray of dynamical spectra (n_nu, n_t).
    :param n_max_components:
        Maximum number of components to check.
    :param frac:
        Integer. Fraction ``1/frac`` will be used for ``sklearn.mixture.DPGMM``
        fitting.

    :return:
        Tuple. First element is dictionary with keys - number of component,
        values - lists of component mean, sigma, number of points & weight.
        Second element is 2D numpy.ndarray with shape as original ``dsp`` shape
        where each pixel has value equal to component it belongs to.
    """
    data = dsp.copy()
    data = data.ravel()[::frac]
    data = data.reshape((data.size, 1))
    clf = DPGMM(n_components=n_max_components, alpha=1)
    clf.fit(data)
    y = clf.predict(data)
    components = sorted(list(set(y)))
    components_dict = {i: [clf.means_[i][0], clf.weights_[i],
                           1./np.sqrt(clf.precs_[i][0]), list(y).count(i)] for
                       i in components}

    # Append weight
    for k, a in components_dict.items():
        components_dict[k].append(a[1] * a[3])
        components_dict[k].remove(a[1])
    sum_nw = sum(a[-1] for a in components_dict.values())
    for k, a in components_dict.items():
        components_dict[k][-1] /= sum_nw

    dsp_classified = y.reshape(dsp.shape)

    return components_dict, dsp_classified


def find_robust_gaussian_params(data):
    """
    Calculate first two moments of gaussian in presence of outliers.
    :param data:
        Iterable of data points.
    :return:
        Mean & std of gaussian distribution.
    """
    mean = biweight_location(data)
    std = mad_std(data)
    return mean, std


def find_file(fname, path='/'):
    """
    Find a file ``fname`` in ``path``. Wildcards are supported
    (ak)
    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, fname):
            matches.append(os.path.join(root, filename))
    if not matches:
        return None
    return matches
