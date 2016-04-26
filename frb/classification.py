import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# TODO: If many objects are found for some time interval - consider RFI
# TODO: Select parameters of training sample (DM, dt, amp, # of samples, # of
# positives)
# TODO: Add CV selection of classifies hyperparameters.
class PulseClassifier(object):
    def __init__(self, clf=SVC, *args, **kwargs):
        self._clf = clf(args, kwargs)
        self._clf_args = args
        self._clf_kwargs = kwargs

    # TODO: I need classifier for different DM ranges. Specify it in arguments.
    def create_train_sample(self, dsp=None, frac=None, dm_range=None,
                            amps=None):
        pass

    def create_test_sample(self, dsp=None, frac=None, dm_range=None, amps=None):
        pass

    def train(self):
        pass

    def test(self):
        pass

    # Should use the same DM values, threshold as for training?
    def classify(self, dsp):
        pass


def plot_2d(i, j, x, y, std=0.01):
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    x = np.atleast_2d(x)
    scaler = StandardScaler().fit(x)
    X_scaled = scaler.transform(x)
    indx = y > 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_scaled[:, i][~indx] + np.random.normal(0, std, np.shape(x)[1])[~indx],
            X_scaled[:, j][~indx] + np.random.normal(0, std, np.shape(x)[1])[~indx],
            '.k')
    ax.plot(X_scaled[:, i][indx] + np.random.normal(0, std, np.shape(x)[1])[indx],
            X_scaled[:, j][indx] + np.random.normal(0, std, np.shape(x)[1])[indx],
            '.r')
    plt.xlabel("Feature {}".format(i))
    plt.ylabel("Feature {}".format(j))


if __name__ == '__main__':
    # Use case with training
    print "Creating Dynamical Spectra"
    from frames import Frame
    frame = Frame(256, 10000, 1684., 0., 16./256, 1./1000)
    n_pulses = 5
    # Step of de-dispersion
    d_dm = 25.
    print "Adding {} pulses".format(n_pulses)
    amps = np.random.uniform(0.1, 0.15, size=n_pulses)
    widths = np.random.uniform(0.001, 0.005, size=n_pulses)
    dm_values = np.random.uniform(0, 1000, size=n_pulses)
    times = np.random.uniform(0., 10., size=n_pulses)
    for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
        frame.add_pulse(t_0, amp, width, dm)
        print "Adding pulse with t0={}, amp={}, width={}, dm={}".format(t_0,
                                                                        amp,
                                                                        width,
                                                                        dm)
    print "Adding noise"
    frame.add_noise(0.5)
    frb_clf = PulseClassifier()
    frb_clf.create_train_sample(frame.values, frac=0.01)
    frb_clf.train()
