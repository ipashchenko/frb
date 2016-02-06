import numpy as np
from scipy.ndimage.measurements import (maximum_position, label, find_objects,
                                        mean, minimum, sum, variance, maximum,
                                        median, center_of_mass)
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from sklearn.svm import SVC
from sklearn.preprocessing import scale
import search


# TODO: Use any FITS-file (part of it) for training.
# TODO: If many objects are found for some time interval - consider RFI
# TODO: Select parameters of training sample (DM, dt, amp, # of samples, # of
# positives)
# TODO: Add CV selection of classifies hyperparameters.
class PulseClassifier(object):
    def __init__(self, clf=SVC, *args, **kwargs):
        self._clf = clf(args, kwargs)
        self._clf_args = args
        self._clf_kwargs = kwargs

    def create_train_sample(self):
        pass

    def train(self):
        pass

    # Should use the same DM values, threshold as for training?
    def classify(self):
        pass


def plot_2d(i, j, x, y, std=0.01):
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    x = np.atleast_2d(x)
    indx = y > 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_scaled[:, i][~indx] + np.random.normal(0, std, len(props))[~indx],
            X_scaled[:, j][~indx] + np.random.normal(0, std, len(props))[~indx],
            '.k')
    ax.plot(X_scaled[:, i][indx] + np.random.normal(0, std, len(props))[indx],
            X_scaled[:, j][indx] + np.random.normal(0, std, len(props))[indx],
            '.r')
    plt.xlabel("Feature {}".format(i))
    plt.ylabel("Feature {}".format(j))


if __name__ == '__main__':
    import numpy as np
    from scipy.ndimage.measurements import (maximum_position, label, find_objects,
                                            mean, minimum, sum, variance, maximum,
                                            median, center_of_mass)
    from scipy.ndimage.morphology import generate_binary_structure
    from skimage.measure import regionprops
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.preprocessing import scale, StandardScaler
    import search

    from frames import Frame, DataFrame
    fname = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch'
    dsp = np.loadtxt(fname).T
    dsp = dsp[:, 3000:]
    n_nu, n_t = np.shape(dsp)
    frame = Frame(n_nu, n_t, 1684, 0., 16./n_nu, 0.001)
    frame.add_values(dsp)
    t0 = np.random.uniform(0, 90, size=25)
    amp = np.random.uniform(0.125, 0.25, size=len(t0))
    width = np.random.uniform(0.0005, 0.005, size=len(t0))
    dm = np.random.uniform(200., 900., size=len(t0))

    dm_min = 0
    dm_max = 1000.
    dm_delta = 20.
    perc = 99.65
    d_t = 0.1
    d_dm = 900


    for pars in zip(t0, amp, width, dm):
        print "Adding pulse wih t0, amp, width, DM = ", pars
        frame.add_pulse(*pars)
    dm_grid = frame.create_dm_grid(dm_min, dm_max, dm_delta=dm_delta)
    tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
    image = tdm_image.copy()
    threshold = np.percentile(image.ravel(), perc)
    image[image < threshold] = 0
    s = generate_binary_structure(2, 2)
    label_image, num_features = label(image, structure=s)
    props = regionprops(label_image, intensity_image=tdm_image)
    # Label image
    print "Found ", len(props), " of regions"

    # Find inserted pulses
    trues = list()
    remove_pulses = list()
    debug = list()
    for (t0_, dm_,) in zip(t0, dm):
        print "Finding injected pulse ", t0_, dm_
        true_ = list()
        for prop in props:
            max_pos, _d_dt, _d_dm = search.max_pos(prop, tdm_image)
            dm__, t_ = max_pos
            t_ *= frame.dt
            dm__ *= dm_delta
            prop.t0 = t_
            prop.dm0 = dm__
            prop.dt = _d_dt
            prop.d_dm = _d_dm
            _d_t_ = abs(t_ - t0_)
            _d_dm_ = abs(dm__ - dm_)
            debug.append([_d_t_, _d_dm_])
            if _d_t_ < d_t and _d_dm_ < d_dm:
                print "Found ", t_, dm__, "area : ", prop.area
                print "index in props ", props.index(prop)
                true_.append(prop)
        # Keep only object with highest area if more then one
        print true_
        if not true_:
            print "Haven't found injected pulse ", t0_, dm_
        try:
            trues.append(sorted(true_, key=lambda x: x.area, reverse=True)[0])
        except IndexError:
            remove_pulses.append([t0_, dm_])

    # Now remove pusles that can't be found
    for (t0_, dm_) in remove_pulses:
        for pars in zip(t0, amp, width, dm):
            t0__, _, _, dm__ = pars
            if t0_ == t0__ and dm_ == dm__:
                print "Removing pulse wih t0, amp, width, DM = ", pars
                frame.rm_pulse(*pars)
    # Again find props
    tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
    image = tdm_image.copy()
    threshold = np.percentile(image.ravel(), perc)
    image[image < threshold] = 0
    s = generate_binary_structure(2, 2)
    label_image, num_features = label(image, structure=s)
    props = regionprops(label_image, intensity_image=tdm_image)
    # Label image
    print "After removing not found pulses found ", len(props), " of regions"

    # Find inserted pulses
    trues = list()
    debug = list()
    for (t0_, dm_,) in zip(t0, dm):
        if list((t0_, dm_)) in remove_pulses:
            continue
        print "Finding injected pulse ", t0_, dm_
        true_ = list()
        for prop in props:
            max_pos, _d_dt, _d_dm = search.max_pos(prop, tdm_image)
            dm__, t_ = max_pos
            t_ *= frame.dt
            dm__ *= dm_delta
            prop.t0 = t_
            prop.dm0 = dm__
            prop.dt = _d_dt
            prop.d_dm = _d_dm
            _d_t_ = abs(t_ - t0_)
            _d_dm_ = abs(dm__ - dm_)
            debug.append([_d_t_, _d_dm_])
            if _d_t_ < d_t and _d_dm_ < d_dm:
                print "Found ", t_, dm__, "area : ", prop.area
                print "index in props ", props.index(prop)
                true_.append(prop)
            # Keep only object with highest area if more then one
        print true_
        if not true_:
            raise Exception("Haven't found injected pulse: {}, {}".format(t0_,
                                                                          dm))
        trues.append(sorted(true_, key=lambda x: x.area, reverse=True)[0])

    # Create arrays with features
    X = list()
    y = list()
    for prop in props:
        X.append([prop.area, prop.dt, prop.d_dm, prop.eccentricity, prop.extent,
                  prop.filled_area, prop.major_axis_length,
                  prop.max_intensity, prop.min_intensity,
                  prop.mean_intensity, prop.orientation, prop.perimeter,
                  prop.solidity])
        if prop in trues:
            print "+1"
            y.append(1)
        else:
            y.append(0)

    # X_scaled = scale(X)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Train classifier
    clf = SVC(kernel='rbf', probability=True, class_weight='auto')
    clf.fit(X_scaled, y)
    # param_grid = {'learning_rate': [0.3, 0.1, 0.05, 0.01],
    #               'max_depth': [3, 4, 6],
    #               'min_samples_leaf': [2, 3, 9, 17],
    #               'max_features': [1.0, 0.3, 0.1]}
    # est = GradientBoostingClassifier(n_estimators=3000)
    # gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_scaled, y)
    # gs_cv.best_params_
    # est = GradientBoostingClassifier(n_estimators=3000, **gs_cv.best_params_)
    # est.fit(X_scaled, y)

    ############################################################################
    # Create some testing data
    print "Creating testing data"
    # fname = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch'
    frame = Frame(n_nu, n_t, 1684, 0., 16./n_nu, 0.001)
    frame.add_values(dsp)
    # frame = DataFrame(fname, 1684., 0., 16. / 128., 0.001)
    t0 = np.random.uniform(0, 90, size=25)
    amp = np.random.uniform(0.125, 0.25, size=len(t0))
    width = np.random.uniform(0.0005, 0.005, size=len(t0))
    dm = np.random.uniform(200., 900., size=len(t0))

    for pars in zip(t0, amp, width, dm):
        print "Adding pulse wih t0, amp, width, DM = ", pars
        frame.add_pulse(*pars)
    dm_grid = frame.create_dm_grid(dm_min, dm_max, dm_delta=dm_delta)
    tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
    image = tdm_image.copy()
    threshold = np.percentile(image.ravel(), perc)
    image[image < threshold] = 0
    s = generate_binary_structure(2, 2)
    label_image, num_features = label(image, structure=s)
    props = regionprops(label_image, intensity_image=tdm_image)
    # Label image
    print "Found ", len(props), " of regions (testing sample)"

    # Add attributes
    for prop in props:
        max_pos, _d_dt, _d_dm = search.max_pos(prop, tdm_image)
        dm__, t_ = max_pos
        t_ *= frame.dt
        dm__ *= dm_delta
        prop.t0 = t_
        prop.dm0 = dm__
        prop.dt = _d_dt
        prop.d_dm = _d_dm
    # Create arrays with features
    X = list()
    for prop in props:
        X.append([prop.area, prop.dt, prop.d_dm, prop.eccentricity, prop.extent,
                  prop.filled_area, prop.major_axis_length,
                  prop.max_intensity, prop.min_intensity,
                  prop.mean_intensity, prop.orientation, prop.perimeter,
                  prop.solidity])
    # X_scaled = scale(X)
    # Use the same scale for transformation
    X_scaled = scaler.transform(X)
    result = clf.predict(X_scaled)
    found = [props[i] for i in np.where(np.asarray(result) > 0)[0]]
    print("Found {} positives".format(len(found)))

    # Print positively classified candidates
    for prop in found:
        max_pos, _d_dt, _d_dm = search.max_pos(prop, tdm_image)
        dm__, t_ = max_pos
        t_ *= frame.dt
        dm__ *= dm_delta
        print "Found ", t_, dm__, "area : ", prop.area

    for t0_real, dm_real in zip(t0, dm):
        found_pulse = False
        for prop in found:
            max_pos, _d_dt, _d_dm = search.max_pos(prop, tdm_image)
            dm__, t_ = max_pos
            t_ *= frame.dt
            dm__ *= dm_delta
            if abs(t0_real - t_) < 0.01 and abs(dm_real - dm__) < 300:
                print("Pulse found: {}, {}".format(t0_real, dm_real))
                found_pulse = True
                break
        if not found_pulse:
            print("Pulse NOT found: {}, {}".format(t0_real, dm_real))


