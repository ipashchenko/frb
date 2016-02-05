import numpy as np
from scipy.ndimage.measurements import (maximum_position, label, find_objects,
                                        mean, minimum, sum, variance, maximum,
                                        median, center_of_mass)
from scipy.ndimage.morphology import generate_binary_structure
from skimage.measure import regionprops
from sklearn.svm import SVC
from sklearn.preprocessing import scale


class PulseClassifier(object):

    @staticmethod
    def max_pos(object, image):
        """
        Returns maximum position and widths in both direction.
        :param object:
        :param image:
        :return:
        """
        subimage = image[object.bbox[0]: object.bbox[2],
                   object.bbox[1]: object.bbox[3]]
        indx = np.unravel_index(subimage.argmax(), subimage.shape)
        return (object.bbox[0] + indx[0], object.bbox[1] + indx[1]), object.bbox[2] - object.bbox[0], object.bbox[3] - object.bbox[1]

    def __init__(self, clf=SVC, *args, **kwargs):
        self._clf = clf(args, kwargs)
        self._clf_args = args
        self._clf_kwargs = kwargs

    def train(self, frame, t0, amp, width, dm, perc, dm_min=0., dm_max=1000.,
              dm_delta=None, d_t=0.005, d_dm=200.):
        """
        Create train data from instance of ``Frame`` subclass.
        :param frame:
        :param t:
        :param amp:
        :param width:
        :param dm:
        :return:
        """
        for pars in zip(t0, amp, width, dm):
            print "Adding pulse wih t0, amp, width = ", pars
            frame.add_pulse(*pars)
        dm_grid = frame.create_dm_grid(dm_min, dm_max, dm_delta=dm_delta)
        if not dm_delta:
            dm_delta = dm_grid[1] - dm_grid[0]
        tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
        threshold = np.percentile(tdm_image.ravel(), perc)
        tdm_image[tdm_image < threshold] = 0
        s = generate_binary_structure(2, 2)
        label_image, num_features = label(tdm_image, structure=s)
        props = regionprops(label_image, intensity_image=tdm_image)
        # Label image
        print "Found ", len(props), " of regions"

        # Find inserted pulses
        trues = list()
        for (t0_, dm_,) in zip(t0, dm):
            print "Finding injected pulse ", t0_, dm_
            true_ = list()
            for prop in props:
                max_pos, _d_dt, _d_dm  = PulseClassifier.max_pos(prop, tdm_image)
                dm__, t_ = max_pos
                t_ *= frame.dt
                dm__ *= dm_delta
                prop.t = t_
                prop.dm = dm__
                _d_t_ = abs(t_ - t0_)
                _d_dm_ = abs(dm__ - dm_)
                if (_d_t_ < d_t and  _d_dm_ < d_dm):
                    print "Found ", t_, dm__, "area : ", prop.area
                    print "index in props ", props.index(prop)
                    true_.append(prop)
            # Keep only object with highest area if more then one
            if not true_:
                print "Haven't found injected pulse ", t0_, dm_
            trues.append(sorted(true_, key=lambda x: x.area, reverse=True)[0])

        # Create arrays with features
        X = list()
        y = list()
        for prop in props:
            X.append([prop.area, _d_dt, _d_dm, prop.eccentricity, prop.extent,
                      prop.filled_area, prop.major_axis_length,
                      prop.max_intensity, prop.min_intensity,
                      prop.mean_intensity, prop.orientation, prop.perimeter,
                     prop.solidity])
            if prop in trues:
                y.append(1)
            else:
                y.append(0)

        X_scaled = scale(X)
        # Train classifier
        self._clf.fit(X_scaled, y)

    def classify(self, frame, perc, dm_min=0., dm_max=1000., dm_delta=None):
        dm_grid = frame.create_dm_grid(dm_min, dm_max, dm_delta=dm_delta)
        if not dm_delta:
            dm_delta = dm_grid[1] - dm_grid[0]
        tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
        threshold = np.percentile(tdm_image.ravel(), perc)
        tdm_image[tdm_image < threshold] = 0
        s = generate_binary_structure(2, 2)
        label_image, num_features = label(tdm_image, structure=s)
        props = regionprops(label_image, intensity_image=tdm_image)
        # Label image
        print "Found ", len(props), " of regions"

        # Create arrays with features
        X = list()
        y = list()
        for prop in props:
            max_pos, _d_dt, _d_dm  = PulseClassifier.max_pos(prop, tdm_image)
            dm__, t_ = max_pos
            t_ *= frame.dt
            dm__ *= dm_delta
            prop.t = t_
            prop.dm = dm__
            X.append([prop.area, _d_dt, _d_dm, prop.eccentricity,
                      prop.extent, prop.filled_area, prop.major_axis_length,
                      prop.max_intensity, prop.min_intensity,
                      prop.mean_intensity, prop.orientation, prop.perimeter,
                      prop.solidity])

        X_scaled = scale(X)
        y = self._clf.fit(X_scaled)
        indxs = np.where(y == 1)[0]
        result = list()
        for indx in indxs:
            result.append(props[indx])

        return result


def timing(frame, dm_delta=None, dm_max=1000.):
    import time
    if dm_delta:
        dm_grid = frame.create_dm_grid(0., dm_max, dm_delta)
        t0 = time.time()
        tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
        t1 = time.time()
        print t1 - t0
        return tdm_image
    else:
        from FDMT import FDMT
        n = int(dm_max / 35.)
        t0 = time.time()
        dmt = FDMT(frame.values[:, :524288], 1668.125, 1684., n,
                   frame.values.dtype)
        t1 = time.time()
        print t1 - t0
        return dmt



if __name__ == '__main__':
    from frames import DataFrame
    fname = '/home/ilya/code/frb/data/crab_600sec_64ch_1ms.npy'
    frame = DataFrame(fname, 1684., 0., 16. / 64., 0.001)
    t0 = range(10, 600, 25)
    amp = np.random.uniform(3., 4., size=len(t0))
    width = np.random.uniform(0.0001, 0.01, size=len(t0))
    dm = np.random.uniform(200., 900., size=len(t0))

    dm_min=0
    dm_max=1000.
    dm_delta = None
    perc = 99.85
    d_t = 0.002
    d_dm = 100


    for pars in zip(t0, amp, width, dm):
        print "Adding pulse wih t0, amp, width = ", pars
        frame.add_pulse(*pars)
    dm_grid = frame.create_dm_grid(dm_min, dm_max, dm_delta=dm_delta)
    if not dm_delta:
        dm_delta = dm_grid[1] - dm_grid[0]
    tdm_image = frame.grid_dedisperse(dm_grid, threads=4)
    threshold = np.percentile(tdm_image.ravel(), perc)
    tdm_image[tdm_image < threshold] = 0
    s = generate_binary_structure(2, 2)
    label_image, num_features = label(tdm_image, structure=s)
    props = regionprops(label_image, intensity_image=tdm_image)
    # Label image
    print "Found ", len(props), " of regions"

    # Find inserted pulses
    trues = list()
    for (t0_, dm_,) in zip(t0, dm):
        print "Finding injected pulse ", t0_, dm_
        true_ = list()
        for prop in props:
            max_pos, _d_dt, _d_dm = PulseClassifier.max_pos(prop, tdm_image)
            dm__, t_ = max_pos
            t_ *= frame.dt
            dm__ *= dm_delta
            prop.t0 = t_
            prop.dm0 = dm__
            prop.dt = _d_dt
            prop.d_dm = _d_dm
            _d_t_ = abs(t_ - t0_)
            _d_dm_ = abs(dm__ - dm_)
            if (_d_t_ < d_t and  _d_dm_ < d_dm):
                print "Found ", t_, dm__, "area : ", prop.area
                print "index in props ", props.index(prop)
                true_.append(prop)
        # Keep only object with highest area if more then one
        print true_
        if not true_:
            print "Haven't found injected pulse ", t0_, dm_
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

    X_scaled = scale(X)
    # Train classifier
    clf = SVC(kernel='rbf', class_weight={1: 10})
    clf.fit(X_scaled, y)


    # classifier = PulseClassifier(SVC, kernel='linear', class_weight={1: 10})
    # classifier.train(frame, t0, amp, width, dm, 99.85, d_t=0.005, d_dm=200.)

    # Create some data
    print "Creating testing data"
    frame = DataFrame(fname, 1684., 0., 16. / 64., 0.001)
    frame.add_pulse(230., 2.1, 0.005, dm=450.)
    frame.add_pulse(130., 2.5, 0.002, dm=650.)
    frame.add_pulse(530., 2.4, 0.003, dm=350.)
    result = clf.predict(X_scaled)
    print result
