# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from skimage.morphology import opening
from candidates import Candidate
from astropy.time import TimeDelta


def find_peaks(array_like, n_std=4, med_width=31, gauss_width=2):
    """
    Find peaks in 1D array.

    Data first median filtered with window size ``med_width``. Then it convolved
    with gaussian filter of width ``gauss_width``. Indexes of data with values
    higher then ``n_std`` are returned.

    :param array_like:
        Iterable of data values.
    :param med_width: (optional)
        Width of median filter to preprocess data. (default: ``30``)
    :param gauss_width: (optional)
        Width of gaussian filter to convolve median filtered data. (default:
        ``2``)
    :param n_std: (optional)
        Number of standard deviations to consider when searching peaks.
        (default: 4)

    :return:
        Numpy array of indexes of peak values.
    """
    import scipy
    array = np.asarray(array_like)
    array = scipy.signal.medfilt(array, med_width)
    garray = scipy.ndimage.filters.gaussian_filter1d(array, gauss_width)
    ind = (garray - np.mean(garray)) > n_std * np.std(garray)
    return ind


def max_pos(object, image):
    """
    Returns maximum position and widths in both direction.

    :param object:
        ``skimage.measure._regionprops._RegionProperties`` instance.
    :param image:
        Original image
    :return:
        Tuple of max position & 2 widths of region.
    """
    subimage = image[object.bbox[0]: object.bbox[2],
               object.bbox[1]: object.bbox[3]]
    indx = np.unravel_index(subimage.argmax(), subimage.shape)
    return (object.bbox[0] + indx[0], object.bbox[1] + indx[1]),\
            object.bbox[2] - object.bbox[0], object.bbox[3] - object.bbox[1]


# TODO: All search functions must returns instances of ``Candidate`` class
def search_candidates_clf(dsp, frb_clf=None, training_frac=0.01):
    """
    Search FRB using ML.
    :param dsp:
        Dynamical spectra.
    :param frb_clf: (optional)
        Instance of ``PulseClassifier``. If ``None`` then initialize one and
        train usign data supplied. (default: ``None``)
    :param training_frac: (optional)
        Fraction of time interval of dynamical spectra used for training
        classifier. (default: ``0.01``)
    :return:
    """
    from classification import PulseClassifier
    if frb_clf is None:
        frb_clf = PulseClassifier()
        frb_clf.create_train_sample(dsp, frac=training_frac)
        frb_clf.train()
    candidates = frb_clf.classify(dsp)
    return candidates


# TODO: All search functions must returns instances of ``Candidate`` class
def search_candidates(image, n_d_x, n_d_y, t_0, d_t, d_dm):

    a = image.copy()
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
    # Find objects
    objects = find_objects(labeled_array)
    props = regionprops(labeled_array, intensity_image=image)
    # Container of object's properties
    _objects = np.empty(num_features, dtype=[('label', 'int'),
                                             ('dx', '<f8'),
                                             ('dy', '<f8'),
                                             ('max_pos', 'int',
                                              (2,))])

    labels = np.arange(num_features) + 1
    dx = [int(obj[1].stop - obj[1].start) for obj in objects]
    dy = [int(obj[0].stop - obj[0].start) for obj in objects]

    # Filling objects structured array
    _objects['label'] = labels
    _objects['dx'] = dx
    _objects['dy'] = dy
    # Classify objects
    _objects = _objects[np.logical_and(_objects['dy'] > n_d_y,
                                       _objects['dx'] > n_d_x)]
    # Fetch positions of only successfuly classified objects
    _objects['max_pos'] = maximum_position(image, labels=labeled_array,
                                           index=_objects['label'])
    _objects = _objects[np.lexsort((_objects['dx'], _objects['dy']))[::-1]]

    candidates = list()
    for _object in _objects:
        max_pos = _object['max_pos']
        candidate = Candidate(t_0 + max_pos[1] * TimeDelta(d_t, format='sec'),
                              max_pos[0] * float(d_dm))
        candidates.append(candidate)

    return candidates


# FIXME: ``skimage.filters.median`` use float images with ranges ``[-1, 1]``. I
# can scale original, use ``median`` and then scale back - it is much faster
# then mine
def create_ellipses(tdm_image, disk_size=5, threshold_perc=99.5,
                    statistic='mean', opening_selem=np.ones((4, 4))):
    """
    Function that pre-process de-dispersed plane `t-DM` by creating
    characteristic inclined ellipses in places where FRB is sitting.

    :param tdm_image:
        2D numpy.ndarray  of `t-DM` plane.
    :param disk_size: (optional)
        Disk size to use when calculating filtered values. (default: ``5``)
    :param threshold_perc: (optional)
        Threshold [0. - 100.] to threshold image after filtering. (default:
        ``99.5``)
    :param statistic: (optional)
        Statistic to use when filtering (``mean``, ``median`` or ``gauss``).
        (default: ``mean``)
    :param opening_selem: (optional)
        The neighborhood expressed as a 2-D array of 1’s and 0’s for opening
        step. (default: ``np.ones((4, 4))``)

    :return:
        2D numpy.ndarray of thresholded image of `t - DM` plane.
    """
    statistic_dict = {'mean': circular_mean, 'median': circular_median,
                      'gauss': gaussian_filter}
    image = tdm_image.copy()
    image = statistic_dict[statistic](image, disk_size)
    threshold = np.percentile(image.ravel(), threshold_perc)
    image[image < threshold] = 0
    image = opening(image, opening_selem)
    return image


def circular_mean(data, radius):
    """
    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf
    from skimage.morphology import disk

    kernel = disk(radius)
    return gf(data, np.mean, footprint=kernel)


def gaussian_filter(data, sigma):
    from skimage.filters import gaussian_filter as gf
    return gf(data, sigma)


def circular_median(data, radius):
    """
    :param data:
    :param radius:
    :return:
    """
    from scipy.ndimage.filters import generic_filter as gf
    from skimage.morphology import disk

    kernel = disk(radius)
    return gf(data, np.median, footprint=kernel)


def get_props(image, threshold):
    """
    Rerurn measured properties list of imaged labeled at specified threshold.

    :param image:
        Numpy 2D array with image.
    :param threshold:
        Threshold to label image. [0.-100.]

    :return:
        List of RegionProperties -
        (``skimage.measure._regionprops._RegionProperties`` instances)
    """
    threshold = np.percentile(image.ravel(), threshold)
    a = image.copy()
    # Keep only tail of image values distribution with signal
    a[a < threshold] = 0
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
    return regionprops(labeled_array, intensity_image=image)

