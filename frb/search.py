import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure


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
