import numpy as np
from itertools import combinations


vround = np.vectorize(round)
vint = np.vectorize(int)


def delta_dm_max(nu_max, nu_min, dt):
    """
    Return difference in DM that corresponds to arrival time shift between
    highest and lowest frequency channels equals to time resolution.

    :param nu_max:
        Frequency of highest frequency channel [MHz].
    :param nu_min:
        Frequency of lowest frequency channel [MHz].
    :param dt:
        Time interval between measurements [s].
    :return:
        Maximum difference of DM that we should use for FRB grid search.

    >>> delta_dm_max(1600., 1600. - 16., 3. / 1000)
    1.8147253416989504

    """
    # MHz ** 2 * cm ** 3 * s / pc
    k = 1. / (2.410331 * 10 ** (-4))
    return dt / (k * (1. / (nu_min) ** 2.) - 1. / nu_max ** 2.)


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


def find_close_2(arr1, arr2, indx_dict):
    """
    Function that search ``similar`` rows in two numpy 2d-arrays using specified
    indexes and allowed differences for that indexes.

    :param arr1:
        First array to compare.
    :param arr2:
        Second array to compare.
    :param indx_dict:
        Dictionary with keys - indexes and values - absolute value of allowed
        differences between 2 arrays for current index.
    :return:
        List of 2d numpy arrays with row1 & row2 - rows from ``arr1`` & ``arr2``
        which are coincide. Empty list if were no coincidence.

    """
    # Check if array has zero size
    if not arr1.size or not arr2.size:
        raise Exception("One of the arrays has zero size")
    arr1 = np.atleast_2d(arr1)
    arr2 = np.atleast_2d(arr2)
    successful_candidates = list()
    for el in arr1:
        diff = arr2 - el
        indxs = list()
        for key, value in indx_dict.items():
            indxs.append(abs(diff[:, key]) < value)
        indxs = np.vstack(indxs).T
        for i, indx in enumerate(indxs):
            if np.alltrue(indx):
                successful_candidates.append(np.vstack((el, arr2[i],)))

    return successful_candidates


def find_close(array_dict, indx_dict):
    """
    Function that find close rows in arrays that are values of user-specified
    dictionary ``array_dict`` and returns list of dictionaries where each
    dictionary's values represent a bunch of close rows and keys denotes what
    input arrays do correspond to which rows.

    :param array_dict:
        Dictionary of 1D numpy arrays.
    :param indx_dict:
        Dictionary of deltas that define `close` values for given index.

    :return:
        List of dictionaries with keys - arrays, values - close rows in this
        arrays.

    """
    # List that will contain dictionaries where each dictionary contains
    # information about single coincidence among several arrays
    results = list()

    for pair in combinations(array_dict.keys(), 2):
        # Reduce the length of for-loop in ``find_close_2``
        if len(array_dict[pair[0]]) < len(array_dict[pair[1]]):
            pair = pair[::-1]
        close_2 = find_close_2(array_dict[pair[0]], array_dict[pair[1]],
                               indx_dict)
        # If no coincidence for this pair of antennas => continue
        if not close_2:
            continue
        # If there's coincidence but no coincidences were before then append
        # dictionaries in ``dicts`` to ``results`` or extend those in
        # ``results`` that have close rows in their arrays.
        dicts = [{pair[0]: result[0], pair[1]: result[1]} for result in
                 close_2]
        if not results:
            results.append(*dicts)
            continue
        # If already were coincidence then loop for past coincidences
        for dict_ in dicts:
            print "dict_"
            print dict_
            # Check all dictionaries in ``results`` list and if some contains
            # close values then extend this dicts. If no such then append
            # ``dict_`` to ``results``.
            do_append = True
            # Iterating over copy cause original will be updated during loop
            for result_dict in results[:]:
                print "result_dict"
                print result_dict
                local_close_2 = find_close_2(np.vstack(dict_.values()),
                                             np.vstack(result_dict.values()),
                                             indx_dict)
                # If no coincidence => check other dictionary in ``for``-loop
                if not local_close_2:
                    continue
                # If there's coincidence => extend ``result`` dictionary
                result_dict.update(dict_)
                do_append = False
            else:
                if do_append:
                    results.append(dict_)

    return results


def unique_rows(a):
    """
    Find unique rows in 2D numpy array.
    see Stackexchange.
    """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]
