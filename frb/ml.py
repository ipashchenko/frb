import numpy as np
from search_candidates import Searcher
from search import get_ellipse_features_for_classification, max_pos
from scipy.ndimage.measurements import (maximum_position, label, find_objects,
                                        mean, minimum, sum, variance, maximum,
                                        median, center_of_mass)
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import scale
import search


# TODO: If many objects are found for some time interval - consider RFI
# TODO: Select parameters of training sample (DM, dt, amp, # of samples, # of
# positives)
# TODO: Add CV selection of classifies hyperparameters.
class PulseClassifier(object):
    """
    Class that describes classification of pulses.

    :param clf:
        Classification class.
    """
    def __init__(self, de_disp_func, preprocess_func, clf=SVC, de_disp_args=[],
                 de_disp_kwargs={}, preprocess_args=[], preprocess_kwargs={},
                 clf_args=[], clf_kwargs={}):
        self._clf = clf(*clf_args, **clf_kwargs)
        self._clf_args = clf_args
        self._clf_kwargs = clf_kwargs
        self.de_disp_func = de_disp_func
        self.de_disp_args = de_disp_args
        self.de_disp_kwargs = de_disp_kwargs
        self.preprocess_func = preprocess_func
        self.preprocess_args = preprocess_args
        self.preprocess_kwargs = preprocess_kwargs

    # TODO: I need classifyer for different DM ranges. Specify it in arguments.
    def create_samples(self, dsp, amps, dms, widths, d_t = 0.1, d_dm = 900):
        """

        :param dsp:
            Dynamical spectra used for training/testing classifier.

        :return:
            Two dictionaries with keys - property objects and values - lists of
            features and responses (0/1).
        """
        # add to real data fake FRBs with specified parameters
        # Find list of ``prop`` objects
        # Identify among ``prop`` objects those that are real FRBs.
        # Check that all injected FRBs are among found real values
        t0s = np.linspace(0., dsp.shape[0], len(amps)+2)[1:-1]
        for pars in zip(t0s, amps, widths, dms):
            print "Adding pulse with pars [t,amp, w, dm] {}".format(pars)
            dsp.add_pulse(*pars)
        searcher = Searcher(dsp.values, dsp.meta_data)
        searcher.de_disperse(self.de_disp_func, *self.de_disp_args,
                             **self.de_disp_kwargs)
        searcher.pre_process(self.preprocess_func, *self.preprocess_args,
                             **self.preprocess_kwargs)
        features = get_ellipse_features_for_classification(searcher._pre_processed_data)
        print "Features ", features

        # Find inserted pulses
        trues = list()
        remove_pulses = list()
        debug = list()
        # FIXME: This assumes that only one positional argument is dm array.
        dm_range = self.de_disp_args[0]
        dm_delta = dm_range[1] - dm_range[0]
        for (t0_, dm_,) in zip(t0s, dms):
            print "Finding injected pulse ", t0_, dm_
            true_ = list()
            for prop in features.keys():
                max_pos_, _d_dt, _d_dm = max_pos(prop,
                                                searcher._de_dispersed_data)
                # print "max_pos, dt, dm for prop: ", max_pos_, _d_dt, _d_dm
                dm__, t_ = max_pos_
                t_ *= dsp.dt
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
                    true_.append(prop)
            # Keep only object with highest area if more then one
            if not true_:
                print "Haven't found injected pulse ", t0_, dm_
            try:
                trues.append(sorted(true_, key=lambda x: x.area,
                                    reverse=True)[0])
            except IndexError:
                remove_pulses.append([t0_, dm_])

        # Now remove pusles that can't be found
        for (t0_, dm_) in remove_pulses:
            for pars in zip(t0s, amps, widths, dms):
                t0__, _, _, dm__ = pars
                if t0_ == t0__ and dm_ == dm__:
                    print "Removing pulse wih t0, amp, width, DM = ", pars
                    dsp.rm_pulse(*pars)

        # Again find props now without pulses that can't be found
        searcher = Searcher(dsp.values, dsp.meta_data)
        searcher.de_disperse(self.de_disp_func, *self.de_disp_args,
                             **self.de_disp_kwargs)
        searcher.pre_process(self.preprocess_func, *self.preprocess_args,
                             **self.preprocess_kwargs)
        features = get_ellipse_features_for_classification(searcher._pre_processed_data)
        print "After removing not found pulses found ", len(features.keys()),\
            " of regions"

        # Find inserted pulses
        trues = list()
        debug = list()
        for (t0_, dm_,) in zip(t0s, dms):
            if list((t0_, dm_)) in remove_pulses:
                continue
            print "Finding injected pulse ", t0_, dm_
            true_ = list()
            for prop in features.keys():
                max_pos_, _d_dt, _d_dm = max_pos(prop,
                                                searcher._de_dispersed_data)
                dm__, t_ = max_pos_
                t_ *= dsp.dt
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
                    true_.append(prop)
                    # Keep only object with highest area if more then one
            if not true_:
                raise Exception("Haven't found injected pulse: {}, {}".format(t0_,
                                                                              dm_))
            trues.append(sorted(true_, key=lambda x: x.area, reverse=True)[0])

        # Create arrays with features
        props_responces = dict()
        for prop, prop_features in features.items():
            if prop in trues:
                props_responces[prop] = 1
            else:
                props_responces[prop] = 0
        return features, props_responces

    def train(self, features_dict, responces_dict):
        X = list()
        y = list()
        for prop in features_dict.keys():
            features = np.array(features_dict[prop])
            if np.any(np.isnan(features)):
                continue
            X.append(features)
            y.append((responces_dict[prop]))

        scaler = StandardScaler().fit(X)
        self.scaler = scaler
        X_scaled = scaler.transform(X)
        self._clf.fit(X_scaled, y)
        # param_grid = {'learning_rate': [0.3, 0.1, 0.05, 0.01],
        #               'max_depth': [3, 4, 6],
        #               'min_samples_leaf': [2, 3, 9, 17],
        #               'max_features': [1.0, 0.3, 0.1]}
        # est = GradientBoostingClassifier(n_estimators=3000)
        # gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_scaled, y)
        # print gs_cv.best_params_
        # est = GradientBoostingClassifier(n_estimators=3000, **gs_cv.best_params_)
        # est.fit(X_scaled, y)
        # pass

    def classify_data(self, dsp):
        """
        Classify some data.

        :param dsp:
            Dynamical spectra used for training/testing classifyer.
        :return:
            Logical numpy.ndarray with classification results.
        """
        searcher = Searcher(dsp.values, dsp.meta_data)
        searcher.de_disperse(self.de_disp_func, *self.de_disp_args,
                             **self.de_disp_kwargs)
        searcher.pre_process(self.preprocess_func, *self.preprocess_args,
                             **self.preprocess_kwargs)
        features_dict = get_ellipse_features_for_classification(searcher._pre_processed_data)

        # Remove regions with ``nan`` features
        for prop in sorted(features_dict.keys()):
            features = np.array(features_dict[prop])
            if np.any(np.isnan(features)):
                del features_dict[prop]

        X = list()
        for prop in sorted(features_dict.keys()):
            features = np.array(features_dict[prop])
            X.append(features)
        X_scaled = self.scaler.transform(X)
        y = self._clf.predict(X_scaled)
        responces_dict = dict()
        for i, prop in enumerate(sorted(features_dict.keys())):
            responces_dict[prop] = y[i]
        return features_dict, responces_dict


def plot_2d(X_scaled, i, j, y, std=0.01):
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    indx = y > 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_scaled[:, i][~indx] + np.random.normal(0, std, len(X_scaled))[~indx],
            X_scaled[:, j][~indx] + np.random.normal(0, std, len(X_scaled))[~indx],
            '.k')
    ax.plot(X_scaled[:, i][indx] + np.random.normal(0, std, len(X_scaled))[indx],
            X_scaled[:, j][indx] + np.random.normal(0, std, len(X_scaled))[indx],
            '.r')
    plt.xlabel("Feature {}".format(i))
    plt.ylabel("Feature {}".format(j))


if __name__ == '__main__':
    from frames import create_from_txt
    txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
    dsp = create_from_txt(txt, 1684., 0, 16./128, 0.001)
    dsp = dsp.slice(0.2, 0.5)
    from astropy.time import Time
    t0 = Time.now()
    dsp.meta_data = {'antenna': 'WB', 'freq': 'L', 'band': 'U', 'pol': 'R',
                     'exp_code': 'raks00', 'nu_max': 1684., 't_0': t0,
                     'd_nu': 16./128., 'd_t': 0.001}
    from search import create_ellipses
    from dedispersion import de_disperse_cumsum
    d_dm = 30.
    dm_grid = np.arange(0., 1000., d_dm)
    pclf = PulseClassifier(de_disperse_cumsum, create_ellipses,
                           de_disp_args=[dm_grid],
                           preprocess_kwargs={'disk_size': 3,
                                              'threshold_big_perc': 97.5,
                                              'statistic': 'mean'},
                           clf_kwargs={'kernel': 'rbf', 'probability': True,
                                       'class_weight': 'balanced'})
    n_pulses = 25
    np.random.seed(123)
    # Generate values of pulse parameters
    amps = np.random.uniform(0.15, 0.25, size=n_pulses)
    widths = np.random.uniform(0.001, 0.005, size=n_pulses)
    dm_values = np.random.uniform(100, 500, size=n_pulses)
    times = np.linspace(0, 20, n_pulses)
    features_dict, responces_dict = pclf.create_samples(dsp, amps, dm_values,
                                                        widths)
    print "Training..."
    pclf.train(features_dict, responces_dict)

    # Make some new data
    dsp = create_from_txt(txt, 1684., 0, 16./128, 0.001)
    dsp = dsp.slice(0.5, 1)
    dsp.meta_data = {'antenna': 'WB', 'freq': 'L', 'band': 'U', 'pol': 'R',
                     'exp_code': 'raks00', 'nu_max': 1684., 't_0': t0,
                     'd_nu': 16./128., 'd_t': 0.001}
    n_pulses = 10
    print "Adding {} real FRBs".format(n_pulses)
    np.random.seed(223)
    # Generate values of pulse parameters
    amps = np.random.uniform(0.15, 0.25, size=n_pulses)
    widths = np.random.uniform(0.001, 0.005, size=n_pulses)
    dm_values = np.random.uniform(100, 500, size=n_pulses)
    times = np.linspace(0.1, dsp.shape[0]-0.1, n_pulses)
    for t, amp, width, dm in zip(times, amps, widths, dm_values):
        dsp.add_pulse(t, amp, width, dm)
    # Classify new data set
    print "Classifying..."
    out_features_dict, out_responces_dict = pclf.classify_data(dsp)
    n_pulses_found = out_responces_dict.values().count(1)
    print "Found {} FRBs".format(n_pulses_found)


