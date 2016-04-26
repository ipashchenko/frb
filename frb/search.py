import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def find_levels_bcp(ts, prob=0.95, p0=0.0001, burnin=500, mcmc=1000, w0=0.2):
    kwargs = {'p0': p0, 'burnin': burnin, 'mcmc': mcmc, 'w0': w0}
    ts = FloatVector(ts)
    # Load library ``bcp``
    bcp = importr("bcp")
    out = bcp.bcp(ts, **kwargs)
    posterior_probs = out.rx2("posterior.prob")
    return np.where(posterior_probs > prob)[0]


def find_levels_regtree(ts):
    """
    Attempt to find regions of high / low signal using decision tree regression.

    If RFI is time-limited then dynamical spectra will show `stripes` of
    enhanced noise. This function will find such stripes.

    :param ts:
        Iterable of signal values.
    :return:
    """
    import scipy
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.grid_search import GridSearchCV
    x = np.arange(len(ts)).reshape((len(ts), 1))

    def get_score(depth, rnd=42):
        max_depth = depth
        x = np.arange(len(ts)).reshape((len(ts), 1))
        # ts = scipy.signal.medfilt(ts, 101)
        # param_grid = {'learning_rate': [0.3, 0.1, 0.05, 0.01],
        #               'max_depth': [3, 4, 6],
        #               'min_samples_leaf': [2, 3, 9, 17],
        #               'max_features': [1.0, 0.3, 0.1]}
        # est = GradientBoostingClassifier(n_estimators=3000)
        # gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_scaled, y)
        # gs_cv.best_params_
        # est = GradientBoostingClassifier(n_estimators=3000, **gs_cv.best_params_)
        # est.fit(X_scaled, y)
        # param_grid = {'max_depth': [2 * i for i in range(1, 100, 10)],
        #               'min_samples_split': [2, 20, 100, 1000]}
        # param_grid = {'min_samples_split': [2, 20, 100, 1000],
        #               'min_samples_leaf': [5, 50, 100, 1000]}
        regr = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=1000,
                                     min_samples_leaf=1000, random_state=rnd,
                                     max_leaf_nodes=max_depth)
        # from sklearn.cross_validation import KFold
        # kf = KFold(len(ts), n_folds=10, shuffle=True)
        # gs_cv = GridSearchCV(regr, param_grid, n_jobs=4, cv=kf).fit(x, ts)
        # gs_cv.best_params_
        # regr = DecisionTreeRegressor(max_depth=max_depth, **gs_cv.best_params_)
        regr.fit(x, ts)
        return regr.score(x, ts)

    def plot_tree(max_depth, rnd=42):
        regr = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=1000,
                                     min_samples_leaf=1000, random_state=rnd,
                                     max_leaf_nodes=max_depth)
        regr.fit(x, ts)
        x_test = np.arange(len(ts)).reshape((len(ts), 1))
        y_test = regr.predict(x_test)
        plt.plot(ts, '.k')
        plt.plot(x_test, y_test)
    # x1 = regr.tree_.threshold[0]
    # y_before, y_after = np.squeeze(regr.tree_.value)[1:]

    scores = np.array([get_score(depth) for depth in range(2, 100)])
    # TODO: Find break that corresponds to big differences.
    plt.plot(range(2, 100), scores)



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


def find_noisy(dsp, n, axis=0, rnd=42):
    """
    Attempt to find noisy parts of dynamical spectra. Fitting gaussian mixture
    model to histogram of ``dsp`` value. If 2 components is superior model based
    on BIC then we can find typical threshold for high-amplitude noise.
    :param dsp:
    :param n:
    :param axis:
    :return:
    """
    from sklearn.mixture import GMM
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
        ts = np.mean(dsp, axis=0)
        import scipy
        tsf = scipy.signal.medfilt(ts, 3)
        dsp_changes = tsf < threshold
        n_peaks = len(np.where(dsp_changes[:-1] != dsp_changes[1:])[0])
        regr = DecisionTreeRegressor(max_depth=n_peaks, min_samples_split=50,
                                     min_samples_leaf=50, random_state=rnd,
                                     max_leaf_nodes=n_peaks)
        x = np.arange(len(ts)).reshape((len(ts), 1))

        regr.fit(x, ts)
        x_test = np.arange(len(ts)).reshape((len(ts), 1))
        y_test = regr.predict(x_test)
        plt.plot(ts, '.k')
        plt.plot(x_test, y_test)


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


def create_ellipses(tdm_image, disk_size=5, threshold_perc=99.8):
    """
    Function that preprocess de-dispersed plane `t-DM` by creating
    characteristic inclined ellipses in places where FRB is sitting.

    :param tdm_image:
    :param disk_size:
    :param threshold_perc:
    :return:
    """
    from skimage.filters import median
    from skimage.morphology import disk
    image = tdm_image.copy()
    med = median(image, disk(disk_size))
    threshold = np.percentile(med.ravel(), threshold_perc)
    med[med < threshold] = 0
    return med


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

