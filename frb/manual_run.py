# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:54:11 2016

@author: osh
"""
import raw_data

m5file = '/home/osh/frb_test/raw_data/rk12ec_ys_no0001'
dspec_params = {'nchan':64, 'dt':1, 'offst':0, 'dur':10, 'outfile':None}
m5file_fmt = "Mark5B-256-4-2"
m5 = raw_data.M5(m5file, fmt=m5file_fmt)
ds = m5.create_dspec(**dspec_params)

# to do this correctly one should know IF-config from CFX-file!
#raw_data.dspec_cat(ds['Dspec_file']

print ds