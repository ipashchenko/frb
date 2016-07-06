# Search Fast Radio Bursts in Radioastron-project archive data

## Installing

```
$ git clone https://github.com/akutkin/frb.git
```
- 

## Finding injected pulses in one file

- download sample data
```
$ cd frb/examples
$ wget https://www.dropbox.com/s/ag7rz88kjnblqzv/data.tgz?
$ tar -cvzf data.tgz
```
- run  script
```
$ python2 frb/examples/caching.py
```
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
