# Search Fast Radio Bursts in Radioastron-project archive data

## Installing

```
$ git clone https://github.com/akutkin/frb.git
```
- Create virtual environment for installing dependencies
```
$ cd frb
$ wget https://github.com/pypa/virtualenv/archive/master.zip
$ unzip master.zip
$ python2 virtualenv-master/virtualenv.py ./venv
$ source venv/bin/activate
```
- Install dependencies (note that ``scipy`` & ``numpy`` are already installed on ``frb`` computer, so you can skip them there)
```
$ pip2 install scipy astropy scikit-learn h5py
```

## Finding injected pulses in one file

- download sample data
```
$ cd examples
$ wget https://www.dropbox.com/s/ag7rz88kjnblqzv/data.tgz
$ tar -xvzf data.tgz
```
- run  script
```
$ python2 caching.py
```
This script will inject pulses in raw data and search for them using two algorithms. Each one begins with non-coherent de-dispersion and pre-processing the resulting ``t-DM`` plane to reduce the noise and exclude some extended regions of atypicaly high amplitude. Blobs of high intensity are found. Next, 2D elliptical gaussians are fitted to regions of individuals blobs in original``t-DM`` plane. First algorithm chooses candidates with auto-selected threshold gaussian amplitudes and some other parameters of gaussians that are specific to narrow dispersed pulses. Second algorithm uses artificially injected pulses to train ``Gradient Boosting Classifier``. It uses features of fitted gaussians as well as numerous blob properties to build desicion surface in features space.


Searching pulses using fitted elliptical gaussians in ``t-DM`` place is much faster then using ``Gradient Boosting Classifier``. It is because later needs training sample to be constructed & analyzed. Also it finds best parameters of
classifier using grid of their values. All these steps (training of classifier) must be done only once for small portion of data. 


Currently, amplitudes of injected pulses in training phase are set by hand. It will be fixed soon by analyzing amplitudes of `noise` pulses in apriori pulse-free small chunk of data.

Script will create ``png`` plots of found candidates in ``t-DM`` plane in ``frb/examples`` directory and dump data on found candidates and data searched in ``frb/frb/frb.db`` ``SQLite`` database.  It can be easily viewed in ``Firefox`` with ``SQLite Manager`` addon.

## Process experiment
- Login to ``frb`` computer with your credientials
- Clone ``frb`` repository (see Installing)
- Download experiment CFX-file
```
$ cd frb/frb
$ wget https://www.dropbox.com/s/8pcmgmed36fo8uy/RADIOASTRON_RAKS12EC_C_20151030T210000_ASC_V1.cfx
```
- Compile ``my5spec`` program for converting raw ``Mk5`` data to txt-format
```
$ cd ../my5spec; /usr/bin/make
```
- Run example
```
$ cd ../frb
$ python2 pipeline.py
```
Results on data searched & pulse candidates can be found in ``frb/frb/frb.db`` ``SQLite`` database.  
