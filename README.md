# Search Fast Radio Bursts in Radioastron-project archive data

## Installing

```
$ git clone https://github.com/akutkin/frb.git
```
- Install dependencies (note that ``scipy`` & ``numpy`` are already installed on ``frb`` computer, so you can skip them there)
```
$ pip2 install --user scipy astropy scikit-learn h5py
```

## Finding injected pulses in one file

- download sample data
```
$ cd frb/examples
$ wget https://www.dropbox.com/s/ag7rz88kjnblqzv/data.tgz
$ tar -xvzf data.tgz
```
- run  script
```
$ python2 caching.py
```
Searching pulses using fitted elliptical gaussians in ``t-DM`` place is much faster then using ``Gradient Boosting Classifier``. It is because later needs training sample to be constructed & analyzed. Also it finds best parameters of
classifier using grid of their values. All these steps (training of classifier) must be done only once for small portion of data. 


Script will create ``png`` plots of found candidates in ``t-DM`` plane in ``frb/examples`` directory and dump data on found candidates and data searched in ``frb/frb/frb.db`` ``SQLite`` database.  It can be easily viewed in ``Firefox`` with ``SQLite Manager`` addon.

## Process experiment
- Clone ``frb`` repository (see Installing)
- Login to ``frb`` computer with your credientials
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
