import numpy as np
from scipy.ndimage.measurements import maximum_position, label, find_objects
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter


class BasicImageObjects(object):
    """
    Abstract class for handling image objects.

    :param image:
        2D numpy array with image.
    :param perc:
        Value of percent of image values to blank while labeling image with
        objects.

    :notes:
        Sometimes we need more features for classification. Currently it uses
        only ``dx``, ``dy`` for that. Should be an option to include others
        (``max``, ``mean``, ...) but in that case calculations of them for all
        labelled regions will take a lot of time (there are thousands of objects
        in my use case). Current implementation allows this option by means of
        ``_classify`` method. It should calculate necessary features for objects
        in ``objects`` array using original image and labeled array passed as
        arguments.

    """
    def __init__(self, image, perc):
        threshold = np.percentile(image.ravel(), perc)
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
        self.objects = _objects
        self._classify(image, labeled_array)
        # Fetch positions of only successfuly classified objects
        self.max_pos = self._find_positions(image, labeled_array)
        self._sort()

    def _find_positions(self, image, labeled_array):
        return maximum_position(image, labels=labeled_array, index=self.label)

    def _sort(self):
        """
        Method that sorts image objects somehow.
        """
        raise NotImplementedError

    def _classify(self, *args, **kwargs):
        """
        Method that selects only image objects with desirable properties.
        You can use any features of ``objects`` attribute for classification.
        Or you can fetch any other features (like ``max`` or ``variance``)
        using passed as arguments to method ``image`` and ``labeled_array``.
        """
        raise NotImplementedError

    def plot(self, image, labels=None):
        """
        Overplot image with found labelled objects,
        """
        pass

    def save_txt(self, fname, *features):
        """
        Save image objects parameters to text file.
        """
        # Check that features are among attributes of class
        # FIXME: it must be attribute (just field or property)
        for a in features:
            if a not in dir(self.__class__):
                raise AttributeError(a)
        values = np.vstack([self.__getattribute__(a) for a in features]).T
        np.savetxt(fname, values)

    @property
    def dx(self):
        """
        Shortcut for widths of objects in x direction in pixels.
        """
        return self.objects['dx']

    @property
    def dy(self):
        """
        Shortcut for widths of objects in y direction in pixels.
        """
        return self.objects['dy']

    @property
    def label(self):
        """
        Shortcut for objects labels.
        """
        return self.objects['label']

    @property
    def max_pos(self):
        """
        Shortcut for positions of maximum for objects along both axis in pixels.
        """
        return self.objects['max_pos']

    @max_pos.setter
    def max_pos(self, max_pos):
        try:
            self.objects['max_pos'] = max_pos
        except ValueError:
            self.objects['max_pos'] = np.empty((0, 2))

    def __len__(self):
        return len(self.objects)


class ImageObjects(BasicImageObjects):
    """
    Class that handles image objects for images with specified x,y -
    coordinates.

    :param x_grid:
        Array-like of values of x-coordinates.

    :param y_grid:
        Array-like of values of y-coordinates.

    """
    def __init__(self, image, x_grid, y_grid, perc):
        # Need this attributes in ``_classify`` method of base class
        # ``__init__``
        self.x_grid = np.asarray(x_grid)
        self.y_grid = np.asarray(y_grid)
        self.x_step = x_grid[1] - x_grid[0]
        self.y_step = y_grid[1] - y_grid[0]
        super(ImageObjects, self).__init__(image, perc)

    def __add__(self, other):
        values = other.objects.copy()
        # Keep each own's numbering to show it later.
        # values['label'] += len(self.objects)
        self.objects = np.concatenate((self.objects, values))
        self._sort()
        return self

    @property
    def d_x(self):
        """
        Shortcut for widths of objects in x direction in units of ``x_grid``.
        """
        return self.objects['dx'] * self.x_step

    @property
    def d_y(self):
        """
        Shortcut for widths of objects in y direction in units of ``y_grid``.
        """
        return self.objects['dy'] * self.y_step

    @property
    def y(self):
        """
        Shortcut for positions of maximum for objects along y axis in units of
        ``y_grid``.
        """
        return self.y_grid[self.max_pos[:, 0]]

    @property
    def x(self):
        """
        Shortcut for positions of maximum for objects along x axis in units of
        ``x_grid``.
        """
        return self.x_grid[self.max_pos[:, 1]]

    @property
    def xy(self):
        """
        Shortcut for positions of maximum for objects along both axis in units
        of ``x_grid`` and ``y_grid``.
        """
        return np.vstack((self.x, self.y)).T


class TDMImageObjects(ImageObjects):
    """
    Class that handles search of FRBs by analyzing image objects in
    (t, DM)-plane.

    :param d_dm: (optional)
        Value of DM spanned by object to count it as candidate for pulse
        [cm^3/pc]. (default: ``100.``)
    :param dt: (optional)
        Value of t spanned by object to count it as candidate for pulse
        [s]. (default: ``0.003``)

    """
    def __init__(self, image, x_grid, y_grid, perc, d_dm=150., dt=0.005):
        # Need this attributes in ``_classify`` method of base class
        # ``__init__``
        self._d_dm = d_dm
        self._dt = dt
        super(TDMImageObjects, self).__init__(image, x_grid, y_grid, perc)

    def _sort(self):
        """
        Method that sorts objects by widths (first - in x-direction, second - in
        y-direction.
        """
        self.objects = self.objects[np.lexsort((self.dx, self.dy))[::-1]]

    def _classify(self, *args, **kwargs):
        """
        Method that selects only candidates which have dimensions >
        ``self._d_dm`` [cm*3/pc] and > ``self._dt`` [s]
        """
        self.objects = self.objects[np.logical_and(self.d_y > self._d_dm,
                                                   self.d_x > self._dt)]


class BatchedTDMIO(object):
    """
    Class that used to analyze image in batches.

    :param image:
        2D-numpy array of image to be analyzed.
    :param x_grid:
        Array-like of grid of coordinates in x-direction.
    :param y_grid:
        Array-like of grid of coordinates in y-direction.
    :param perc:
        Value of percent of image values to blank while labeling image with
        objects.
    :param d_dm: (optional)
        Value of DM spanned by object to count it as candidate for pulse
        [cm^3/pc]. (default: ``100.``)
    :param dt: (optional)
        Value of t spanned by object to count it as candidate for pulse
        [s]. (default: ``0.003``)

    """
    def __init__(self, image, x_grid, y_grid, perc, d_dm=100., dt=0.003,
                 std=None):
        self.image = np.asarray(image)
        if std is not None:
            self.image = gaussian_filter(self.image, std)
        self.x_grid = np.array(x_grid)
        self.y_grid = np.array(y_grid)
        self.perc = perc
        self.d_dm = d_dm
        self.dt = dt

    def analyze_slice(self, x_slice):
        """
        Method that search for candidates in part of image that is specified by
        slice.
        :param x_slice:
            Slice object that used to choose region of image to be analyzed in
            x-direction.
        :return:
            2D-numpy arrays (# of candidates, 2,) with t[s], DM[cm^3/pc] values
            for each candidate.

        """
        tdmio = TDMImageObjects(self.image[:, x_slice], self.x_grid[x_slice],
                                self.y_grid, self.perc, d_dm=self.d_dm,
                                dt=self.dt)
        return tdmio.xy

    def run(self, batch_size=500000):
        """

        :param batch_size: (optional)
            Size of image in x-direction, that will be searched for objects in
            batches. If ``None`` then don't use batches and search the whole
            image at once. (default: ``None``)
        :return:
            2D-numpy arrays (# of candidates, 2,) with t[s], DM[cm^3/pc] values
            for each candidate.

        """
        length = self.image.shape[1]
        n = length // batch_size + 1
        ends = np.linspace(0, length, n)
        slice_list = [slice(ends[i], ends[i + 1]) for i in range(n - 1)]

        xy = map(self.analyze_slice, slice_list)
        xy = np.vstack(xy)

        return xy
