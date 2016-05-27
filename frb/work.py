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

dspec_params = {'nchan':64, 'dt':1, 'offst':0, 'dur':10, 'outfile':None}
split_duration = 0.5 # split an M5-file into [sec] intervals


### INPUT CODE
code = 'raks12ac'

### CFX Processing
cfx_files = cfx.get_cfx(cfx_path, code)
print("Found {} CFX files".format(len(cfx_files)))
print cfx_files
#raise SystemExit('planned')

### MAIN LOOP:
for cfile in cfx_files:
    cobj = cfx.CFX(cfile)
    cfx_data = cobj.parse_cfx(code)
    if cobj.freq == 'K':
        print("Skipping K-freq CFX file: {}".format(os.path.basename(cfile)))
        print("NOTE: You can delete following files from data path:")
        print(cfx_data)
        continue
    for fname, params in cfx_data.items():
            m5file = find_file(fname, data_path)

            if m5file is None:
                print("main: Can't find file: {}".format(fname))
                continue
#            m5file = ['/home/osh/frb_test/raw_data/new_folder/Pa_rk02ay_test.m5b']
            if len(list(m5file)) > 0:
                print("main: Found more than one file \
                       matching ({}). Taking 1st one".format(fname))
            m5file = m5file[0]
            m5file_fmt = params[2] # Raw data format
            cfx_fmt = params[-1]   # Rec configuration
            m5 = raw_data.M5(m5file, m5file_fmt)
            offst = 0
            dspec_params.update({'dur':split_duration})
            while offst*32e6 < m5.size:
                dspec_params.update({'offst':offst})
#                print dspec_params
                ds = m5.create_dspec(**dspec_params)

                # NOTE: all 4 channels are stacked forming dsarr:
                dsarr = raw_data.dspec_cat(os.path.basename(ds['Dspec_file']),
                                           cfx_fmt)
                metadata = ds
                metadata['Raw_data_file'] = fname
                metadata['Exp_data'] = params
                break
#                plt.figure()
#                plt.imshow(dsarr, aspect='auto')
                print "BRV SEARCHING..."  # search brv in array here

# TODO: save search results, delete data, ...
                offst = offst + split_duration

