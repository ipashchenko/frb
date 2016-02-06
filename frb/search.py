import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops

# TODO: add algorithm in 1D - just searching peaks in freq. averaged
# de-dispersed auto-spectra.


def find_peaks(array, nstd=4, med_width=30, gauss_width=2):
    import scipy
    array = scipy.signal.medfilt(array, med_width)
    garray = scipy.ndimage.filters.gaussian_filter1d(array, gauss_width)
    ind = garray[garray > nstd * np.std(garray)]
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


def search_candidates(image, threshold, n_d_x, n_d_y):

    threshold = np.percentile(image.ravel(), threshold)
    a = image.copy()
    # Keep only tail of image values distribution with signal
    a[a < threshold] = 0
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

    return _objects


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

