import numpy as np
import os
import fnmatch
from sklearn.mixture import DPGMM
from astropy.stats import mad_std, biweight_location
from scipy.stats import norm

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
        Dictionary with keys - number of component, values - lists of component
        mean, sigma, number of points & weight. And 2D numpy.ndarray with shape
        as original ``dsp`` shape where each pixel has value equal to component
        it belongs to.

    :note:
        Noise values are supposed to be contained out of main component (that is
        supposed to contain real signal).
    """
    data = dsp.copy()
    # n_freq = dsp.shape[0]
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

    # components_by_n = sorted(components_dict,
    #                          key=lambda x: components_dict[x][-1],
    #                          reverse=True)
    # components_by_amp = sorted(components_dict,
    #                            key=lambda x: components_dict[x][0])
    # main_component = components_by_n[0]
    # high_noise_components =\
    #     components_by_amp[components_by_amp.index(main_component) + 1:]
    # low_noise_components = \
    #     components_by_amp[:components_by_amp.index(main_component)]

    # high_ranges = dict()
    # low_ranges = dict()
    # n_sigma_main = norm.ppf(1 -
    #                         float(n_freq)/components_dict[main_component][2])
    # high_range_main = components_dict[main_component][0] +\
    #     n_sigma_main * components_dict[main_component][1]
    # low_range_main = components_dict[main_component][0] - \
    #     n_sigma_main * components_dict[main_component][1]

    # for component in high_noise_components:
    #     n_sigma = norm.ppf(1 - float(n_freq)/components_dict[component][2])
    #     low_range = components_dict[component][0] -\
    #         n_sigma * components_dict[component][1]
    #     if low_range > high_range_main:
    #         low_ranges.update({component: low_range})

    # for component in low_noise_components:
    #     n_sigma = norm.ppf(1 - float(n_freq)/components_dict[component][2])
    #     high_range = components_dict[component][0] +\
    #         n_sigma * components_dict[component][1]
    #     if high_range < low_range_main:
    #         high_ranges.update({component: high_range})

    # try:
    #     low_threshold_component = sorted(high_ranges, key=lambda x: high_ranges[x],
    #                                      reverse=True)[0]
    # except IndexError:
    #     low_threshold_component = None
    # try:
    #     high_threshold_component = sorted(low_ranges, key=lambda x: low_ranges[x])[0]
    # except IndexError:
    #     high_threshold_component = None

    # if high_threshold_component is not None:
    #     n_sigma = norm.ppf(1 -
    #                        float(n_freq)/components_dict[high_threshold_component][2])
    #     high_threshold = components_dict[high_threshold_component][0] -\
    #         n_sigma * components_dict[high_threshold_component][1]
    return components_dict, dsp_classified


def find_robust_gaussian_params(data):
    mean = biweight_location(data)
    std = mad_std(data)
    return mean, std


def find_file(fname, path = '/'):
    """
    Find a file (fname) in (path). Wildcards are supported
    (ak)
    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, fname):
            matches.append(os.path.join(root, filename))
    if len(matches)==0:
#        print("find_file: Can't find file ({})".format(fname))
        return None
    return matches
