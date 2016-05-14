import numpy as np
import os
import fnmatch
from sklearn.mixture import GMM

vround = np.vectorize(round)
vint = np.vectorize(int)


def find_noisy(dsp, n):
    """
    Attempt to characterize the noise of dynamical spectra. Fitting gaussian
    mixture model to histogram of ``dsp`` values. If 2 (or more) components is
    superior model based on BIC then we can find typical threshold for
    high-amplitude noise.

    :param dsp:
        2D numpy.ndarray of dynamical spectra.
    :param n:
        Maximum number of components to check.

    :return:
        Number of components.
    """
    results = dict()
    data = dsp.reshape((dsp.size, 1))
    for i in range(1, n + 1):
        classif = GMM(n_components=i)
        classif.fit(data)
        results.update({classif.bic(data): [i, classif]})
    min_bic = min(results.keys())
    i, clf = results[min_bic]
    if i == 2:
        threshold = (clf.means_ + 3. * np.sqrt(clf.covars_))[0][0]
    return i


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
