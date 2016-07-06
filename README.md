# Search Fast Radio Bursts in Radioastron-project archive data

## Installing

```
$ git clone https://github.com/akutkin/frb.git
```
- Install dependencies

## Finding injected pulses in one file

- download sample data
```
$ cd frb/examples
$ wget https://www.dropbox.com/s/ag7rz88kjnblqzv/data.tgz
$ tar -xvzf data.tgz
```
- run  script
```
$ python2 frb/examples/caching.py
```
Searching pulses using fitted elliptical gaussians in ``t-DM`` place is much faster then using ``Gradient Boosting Classifier``. It is because later needs training sample to be constructed & analyzed. Also it finds best parameters of
classifier using grid of their values. All these steps (training of classifier) must be done only once for small portion of data. 


Script will create ``png`` plots of found candidates in ``t-DM`` plane in ``frb/examples`` directory and dump data on found candidates and data searched in ``frb/frb/frb.db`` ``SQLite`` database.  

## Process experiment

- Login to ``frb`` computer with your credientials
- Download experiment CFX-file
```
wget https://www.dropbox.com/s/8pcmgmed36fo8uy/RADIOASTRON_RAKS12EC_C_20151030T210000_ASC_V1.cfx
```

- Run example
```
$ python2 frb/frb/pipeline.py
```
