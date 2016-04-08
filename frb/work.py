# -*- coding: utf-8 -*-
"""
Main data parsing script
Created on Wed Apr  6 10:47:42 2016
@author: osh
"""
import os

import cfx
from utils import find_file
import raw_data

#import config
cfx_path = '/home/osh/frb_test/cfx'
data_path = '/home/osh/frb_test/raw_data'

dspec_len = 1
dspec_params = {'nchan':64, 'dt':1, 'offst':0, 'dur':10.1, 'outfile':None}

split_duration = 0.5 # split an M5-file into [sec] intervals

### INPUT CODE
code = 'raks02ay'

### CFX Processing
cfx_files = cfx.get_cfx(cfx_path, code)
print("Found {} CFX files".format(len(cfx_files)))
print cfx_files
#raise SystemExit('planned')

### MAIN LOOP:
for cfile in cfx_files:
    cobj = cfx.CFX(cfile)
    cfx_data = cobj.parse_cfx(code)
    if cobj.band() == 'K':
        print("Skipping K-band CFX file: {}".format(os.path.basename(cfile)))
        print("NOTE: You can delete following files from data path:")
        print(cfx_data)
        continue
    for fname, params in cfx_data.items():
            m5file = find_file(fname, data_path)

            if m5file is None:
#                print("main: Can't find file: {}".format(fname))
                continue
            if len(m5file) > 1:
                print("main: Found more than one file \
                       matching ({}). Taking 1st one".format(fname))
            m5file = m5file[0]
            m5file_fmt = params[2] # Raw data format
            cfx_fmt = params[-1]   # Rec configuration
            m5 = raw_data.M5(m5file, m5file_fmt)
            offst = 0
            while offst*32e6 < m5.size:
                dspec_params.update({'offst':offst})
                print dspec_params

                ds = m5.create_dspec(**dspec_params)
                print ds
                dsarr = raw_data.dspec_cat(ds['Output'], cfx_fmt)
                print "BRV SEARCHING..."  # search brv in array here
# TODO: save search results, delete data, ...
                offst = offst + split_duration



### M5file (kostyl)

#m5file = '../data/Pa_rk02ay_test.m5b'
#print("Processing data file: {}".format(m5file))

### Create dyn spec (DS) file(s) + metadata
#m5 = raw_data.M5(m5file)
#m5.show_m5info()
#inf = m5.create_dspec(**dspec_params)
#inf

### concatenate dynamical spectra -- OK!
#fname = os.path.basename(m5file)
#fname = cfx_data.keys()[0]
#cfx_fmt = cfx_data.values()[0][-1]
# For Parkes:
#cfx_fmt = ['1668.00-R-L', '1668.00-R-U', '1668.00-L-L', '1668.00-L-U']
#a = raw_data.dspec_cat(fname, cfx_fmt)
#plt.imshow(a, aspect='auto')
#plt.colorbar()
#start_time = m5.get_start_time()

# TODO: process data file )))



