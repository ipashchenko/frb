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
import time
import re
from astropy.time import Time


my5spec = "../my5spec/./my5spec"


class M5(object):
    """ working with raw data """
    def __init__(self, m5file, fmt=None):
        self.m5file = m5file
        self.fmt = fmt
        if self.fmt is None:
            self.fmt = "Mark5B-256-4-2"
            print "WARNING: fmt is not set"
            raise Exception('fmt is not set')
        self.my5spec = my5spec
        self.m5dir = os.path.dirname(os.path.abspath(self.m5file))
        self.size = os.path.getsize(self.m5file)
        self.starttime = self.start_time

    @property
    def start_time(self):
        """ Determine start time for the m5file """
        cmd = "m5time " + self.m5file + " " + self.fmt
        res = subprocess.check_output(cmd.split())
        res = re.search('\d{5}/\d{2}:\d{2}:\d{2}\.\d{2}', res).group()
        m5_mjd = float(res.split('/')[0])
        m5_hms = res.split('/')[1].split(':')
        m5_time = float(m5_hms[0])/24.0 + float(m5_hms[1])/1440.0\
            + float(m5_hms[2])/86400.0
        res = m5_mjd + m5_time
        return Time(res, format='mjd')

    def __repr__(self):
        """ Show some info about the m5file """
        outprint = "File: %s\n" % self.m5file
        outprint += "Format: %s\n" % self.fmt
        outprint += "File size: %s\n" % self.size
        outprint += "File start MJD/time: %s\n" % self.starttime
        outprint += "Last modified: %s\n" %\
                    time.ctime(os.path.getmtime(self.m5file))
        return outprint

    def show_aspec(self, t0=0, nchan=128, nusr=125, chid=2):
        """ Plot Autospectrum with m5spec util for t0 sec in m5file """
        offset = 3.2e7*t0  # 1sec (?) offset step ???
        tmpfile = self.m5dir + "/tmp_aspec"
        cmd = """m5spec -nopol %s %s %s %s %s %s""" % (self.m5file, self.fmt,
                                                       nchan, nusr, tmpfile,
                                                       offset)
        subprocess.call(cmd.split())
        dat = np.loadtxt(tmpfile)
        plt.figure()
        plt.plot(dat[:, 0], dat[:, chid])
        plt.show()
        os.remove(tmpfile)

    def create_dspec(self, n_nu=64, d_t=1, offset=0, dur=None, outfile=None,
                     dspec_path=None, **kwargs):
        """
        Create 4 DS files for selected M5datafile with nchan, dt[ms], ...
        The input options are the same as for my5spec
        """
        if dspec_path is None:
            dspec_path = os.getcwd()

        # my5spec options:
        opt1 = "-a %s " % d_t
        opt2 = "-n %s " % n_nu

        if dur is not None:
            opt3 = "-l %s " % dur
        else:
            opt3 = ""

        if offset != 0.0:
            opt4 = "-o %s " % offset
        else:
            opt4 = ""

        opts = opt1 + opt2 + opt3 + opt4

        if not outfile:
            opts2 = re.sub("-", "", "".join(opts.split()))
            outfile = os.path.join(dspec_path,
                                   os.path.basename(self.m5file).split('.')[0] +
                                   '_' + opts2 + "_dspec")

        cmd = self.my5spec + " " + opts + "%s %s %s"\
                                          % (self.m5file, self.fmt, outfile)
        subprocess.check_call(cmd.split())
        res = {'Dspec_file': outfile}

        return res


# extra manipulations with dspec files
def get_cfx_format(fname, cfx_data):
    return cfx_data[fname][-1]


def dspec_cat(fname, cfx_fmt, pol=True, uplow=True, dspec_path=None):
    """
    Concatenate dynamical spectra files, returning array.
    INPUTS:
    fname - base filename pattern of DS-files\n
    cfx_format - ['4828.00-L-U', '4828.00-R-U', '4828.00-L-L', '4828.00-R-L']\n
    pol - sum polarizations\n
    uplow - concat UPPER and LOWER bands\n
    OUTPUT: np.array
    """
    if dspec_path is None:
        dspec_path = os.getcwd()
    from utils import find_file
    flist = find_file(fname + '*_0?', dspec_path)
    if flist is None:
        raise Exception("dspec_cat: Can't find files matching %s" % fname)
    if len(flist) > len(cfx_fmt):
        raise Exception("WARNING! dspec_cat: There are difference in files"
                        " number and CFX-format length")
    # FIXME: improve the above checkings
    flist = sorted(flist, key=lambda x: int(x[-2:]))
    ashape = np.loadtxt(flist[0]).shape
    arr = np.zeros((ashape[0], ashape[1]*2))
    for fmt, fil in zip(cfx_fmt, flist):
        if fmt.split('-')[2] == 'L':
            arr[:, :ashape[1]] = arr[:, :ashape[1]] + np.loadtxt(fil)[:, ::-1]
        else:
            arr[:, ashape[1]:] = arr[:, ashape[1]:] + np.loadtxt(fil)
    return arr/2
