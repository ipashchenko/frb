import numpy as np
import os
import fnmatch

vround = np.vectorize(round)
vint = np.vectorize(int)


def find_noisy(frame, n, axis=0):
    values = np.mean(frame.values, axis=axis)
    from sklearn.mixture import GMM
    result_dict = {}
    data = values.reshape((values.size, 1))
    for i in range(1, n + 1):
        classif = GMM(n_components=i)
        classif.fit(data)
        result_dict.update({i: [classif.bic(data), classif]})
    return result_dict


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
