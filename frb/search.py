# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from skimage.morphology import opening
from candidates import Candidate
from astropy.time import TimeDelta
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt


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
# FIXME: It uses processed image for searching maximum position. Should it be
# the option to use original image (more variance - less bias)
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


def search_candidates_ell(image, x_stddev, y_to_x_stddev, theta_lims, t_0, d_t,
                          d_dm):
    a = image.copy()
    s = generate_binary_structure(2, 2)
    # Label image
    labeled_array, num_features = label(a, structure=s)
    # Find objects
    props = regionprops(labeled_array, intensity_image=image)
    candidates = list()
    for i, prop in enumerate(props):
        try:
            gg = fit_elliplse(prop, plot=True, show=True, close=True,
                              save_file="search_ell_{}.png".format(i))
        # TODO: Subclass Exception for this case
        except:
            print "2D gaussian fitting failed!"
            continue
        if ((abs(gg.x_stddev) > abs(x_stddev)) and
                (abs(gg.y_stddev / gg.x_stddev) < y_to_x_stddev) and
                (theta_lims[0] < np.rad2deg(gg.theta) % 180 < theta_lims[1])):
            max_pos = (gg.x_mean + prop.bbox[0], gg.y_mean + prop.bbox[1])
            candidate = Candidate(t_0 + max_pos[1] * TimeDelta(d_t,
                                                               format='sec'),
                                  max_pos[0] * float(d_dm))
            candidates.append(candidate)

    return candidates


# FIXME: ``skimage.filters.median`` use float images with ranges ``[-1, 1]``. I
# can scale original, use ``median`` and then scale back - it is much faster
# then mine
def create_ellipses(tdm_image, disk_size=5, threshold_perc=99.5,
                    statistic='mean', opening_selem=np.ones((3, 3))):
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


def fit_elliplse(prop, plot=False, save_file=None, colorbar_label=None,
                 close=False, show=True):
    """
    Function that fits 2D ellipses to `t-DM` image.

    :param prop:
        ``skimage.measure._regionprops._RegionProperties`` instance

    :return:
        Instance of ``astropy.modelling.functional_models.Gaussian2D`` class
        fitted to `t-DM` image in region of ``prop``.
    """
    data = prop.intensity_image.copy()
    # Remove high-intensity background
    try:
        data -= np.unique(sorted(data.ravel()))[1]
    except IndexError:
        raise Exception("No intensity in region!")
    data[data < 0] = 0
    amp, x_0, y_0, width = infer_gaussian(data)
    x_lims = [0, data.shape[0]]
    y_lims = [0, data.shape[1]]
    g = models.Gaussian2D(amplitude=amp, x_mean=x_0, y_mean=y_0,
                          x_stddev=0.5 * width, y_stddev=0.5 * width,
                          theta=0, bounds={'x_mean': x_lims, 'y_mean': y_lims})
    fit_g = fitting.LevMarLSQFitter()
    x, y = np.indices(data.shape)
    gg = fit_g(g, x, y, data)
    print gg.x_stddev, gg.y_stddev
    print abs(gg.x_stddev), abs(gg.y_stddev / gg.x_stddev),\
        np.rad2deg(gg.theta) % 180

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        im = ax.matshow(data, cmap=plt.cm.jet)
        model = gg.evaluate(x, y, gg.amplitude, gg.x_mean, gg.y_mean,
                            gg.x_stddev, gg.y_stddev, gg.theta)
        ax.contour(y, x, model, colors='w')
        ax.set_xlabel('t steps')
        ax.set_ylabel('DM steps')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.00)
        cb = fig.colorbar(im, cax=cax)
        if colorbar_label is not None:
            cb.set_label(colorbar_label)
        if save_file is not None:
            fig.savefig(save_file, bbox_inches='tight', dpi=200)
        if show:
            fig.show()
        if close:
            plt.close()

    return gg


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


def infer_gaussian(data):
    """
    Return (amplitude, x_0, y_0, width), where width - rough estimate of
    gaussian width
    """
    amplitude = data.max()
    x_0, y_0 = np.unravel_index(np.argmax(data), np.shape(data))
    row = data[x_0, :]
    column = data[:, y_0]
    x_0 = float(x_0)
    y_0 = float(y_0)
    dx = len(np.where(row - amplitude/2 > 0)[0])
    dy = len(np.where(column - amplitude/2 > 0)[0])
    width = np.sqrt(dx ** 2. + dy ** 2.)

    return amplitude, x_0, y_0, width


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

