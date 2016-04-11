# -*- coding: utf-8 -*-
"""
Process raw data functions
Created on Thu Apr  7 10:08:02 2016

@author: osh
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
# import fnmatch
import time
import re



# from astropy.io import fits
# from astropy.time import Time

my5spec = "../my5spec/./my5spec"
dspec_path = '/home/osh/frb_test/dspec'

class M5(object):
    """ working with raw data """
# TODO:
# 1. get file format
# 2. by format get start times
# 3. add chan_id when parsing (Petya)
# 4. add scanning for new files
# 5. output Dspec to FITS (?)

    def __init__(self, m5file, fmt = None):
        self.m5file = m5file
        self.fmt = fmt
        if self.fmt is None:
            self.fmt = "Mark5B-256-4-2"
            print "WARNING: fmt is not set. Trying Mark5B-256-4-2"
        self.my5spec = my5spec
        self.m5dir = os.path.dirname(os.path.abspath(self.m5file))
        self.size = os.path.getsize(self.m5file)
        self.starttime = self.get_start_time()
    def get_start_time(self):
        """ Determine start time for the m5file """
        cmd = "m5time " + self.m5file + " " + self.fmt
        try:
            res = subprocess.check_output(cmd.split())
            res = re.search('\d{5}/\d{2}:\d{2}:\d{2}\.\d{2}', res).group()
            m5_mjd = float(res.split('/')[0])
            m5_hms = res.split('/')[1].split(':')
            m5_time = float(m5_hms[0])/24.0 + float(m5_hms[1])/1440.0 \
                      + float(m5_hms[2])/86400.0
            res =  m5_mjd + m5_time
        except:
            res = 0.0
        return res

    def show_m5info(self):
        """ Show some info about the m5file """
        print "File: %s" % self.m5file
        print "Format: %s" % self.fmt
        print "File size: %s" % self.size
        print "File start MJD/time: %s" % self.starttime
        print "Last modified: %s" % time.ctime(os.path.getmtime(self.m5file))

    def show_aspec(self, t0 = 0, nchan = 128, nusr = 125, chid = 2):
        """ Plot Autospectrum with m5spec util for t0 sec in m5file """
        offset = 3.2e7*t0  # 1sec (?) offset step ???
        tmpfile = self.m5dir + "/tmp_aspec"
        cmd = """m5spec -nopol %s %s %s %s %s %s
            """ % (self.m5file, self.fmt, nchan, nusr, tmpfile, offset)
        subprocess.call(cmd.split())
        dat = np.loadtxt(tmpfile)
        plt.figure()
        plt.plot(dat[:,0], dat[:,chid])
        plt.show()
        os.remove(tmpfile)

    def create_dspec(self, nchan=64, dt=1, offst=0, dur=None, outfile=None):
        """
        Create 4 DS files for seleceted M5datafile with nchan, dt[ms], ...
        The input options are the same as for my5spec
        """
# my5spec options:
        opt1 = "-a %s " % dt
        opt2 = "-n %s " % nchan
        if dur is not None: opt3 = "-l %s " % dur  # duration (time limit)
        else: opt3 = ""
        if offst != 0.0: opt4 = "-o %s " % offst # offset (see my5spec)
        else: opt4 = ""
        opts = opt1 + opt2 + opt3 + opt4
        if not outfile:
            opts2 = re.sub("-", "", "".join(opts.split()))
            outfile = os.path.join(dspec_path,
            os.path.basename(self.m5file).split('.')[0] +'_'+ opts2 + "_dspec")

# Usage: "my5spec [-a aver_time] [-n nchan] [-l time_limit] [-o offset]
#         INFILE FORMAT OUTFILE"
        cmd = self.my5spec + " " + opts + "%s %s %s"\
        % (self.m5file, self.fmt, outfile)
        subprocess.check_call(cmd.split())
        ds_start = self.get_start_time() + offst/86400.0

        res = {'Nchan':nchan, 'DT_ms':dt,
               'Start_mjd':ds_start,
               'Duration_sec':dur,
               'Dspec_file':outfile}
        return res

### extra manipulations with dspec files
def get_cfx_format(fname, cfx_data):
    return cfx_data[fname][-1]

def dspec_cat(fname, cfx_fmt, pol=True, uplow=True):
    """
    Concatenate dynamical spectra files, returning array.
    INPUTS:
    fname - base filename pattern of DS-files\n
    cfx_format - ['4828.00-L-U', '4828.00-R-U', '4828.00-L-L', '4828.00-R-L']\n
    pol - sum polarizations\n
    uplow - concat UPPER and LOWER bands\n
    OUTPUT: np.array
    """
    from utils import find_file
    flist = find_file(fname + '*_0?', dspec_path)
    if flist is None:
        print "dspec_cat: Can't find files mutching %s" % fname
        return
    if len(flist) > len(cfx_fmt):
        print "WARNING! dspec_cat: There are difference in files number \
                 and CFX-format length. Taking 1st 4 files:"
        flist = flist[:4]
# FIXME: improve the above checkings
    flist = sorted(flist, key=lambda x: int(x[-2:]))
    ashape = np.loadtxt(flist[0]).shape
    arr = np.zeros((ashape[0], ashape[1]*2))
    for fmt,fil in zip(cfx_fmt, flist):
        if fmt.split('-')[2] == 'L':
            arr[:,:ashape[1]] = arr[:,:ashape[1]] + np.loadtxt(fil)[:,::-1]
        else:
            arr[:,ashape[1]:] = arr[:,ashape[1]:] + np.loadtxt(fil)
    return arr/2






