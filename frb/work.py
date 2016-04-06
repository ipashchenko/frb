# -*- coding: utf-8 -*-
"""
Main data parsing script
Created on Wed Apr  6 10:47:42 2016
@author: osh
"""
import os
from utils import find_file
#import config
cfx_path = '/home/osh/frb_test/cfx'
data_path = '/home/osh/frb_test/raw_data'


import cfx

### INPUT CODE
code = 'raes03gb'

### CFX Processing
cfx_files = cfx.get_cfx(cfx_path, code)
print("Found {} CFX files".format(len(cfx_files)))
print cfx_files
for cfile in cfx_files:
    cobj = cfx.CFX(cfile)
    cfx_data = cobj.parse_cfx(code)
    if cobj.band() == 'K':
        print("Skipping K-band CFX file: {}".format(os.path.basename(cfile)))
        print("NOTE: You can delete following files from data path:\n")
        print(cfx_data.keys())
        continue
    for fname in cfx_data.keys():
        m5file = find_file(fname, data_path)[0]
        if not os.path.exists(m5file):
            print("Can't find M5file {}".format(fname))
            continue
        print("Processing data file: {}".format(m5file))

### M5file (kostyl)

m5file = 'data/Pa_re02ay_test.m5b'


# TODO: process data file )))
#        try:
#            print
#        except IOError:


