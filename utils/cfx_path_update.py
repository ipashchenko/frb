# -*- coding: utf-8 -*-
"""
A script to update local CFX path
Created on Mon Apr 11 11:36:55 2016

@author: osh
"""

import os
import ftputil
import fnmatch
import netrc

cfx_path = '/home/osh/frb_test/cfx' # Directory where the files needs to be downloaded to
pattern = 'RADIOASTRON*.cfx' #filename pattern for what the script is looking for
server = 'archive.asc.rssi.ru'
auth = netrc.netrc().authenticators(server)
login = auth[0]
pwd = auth[2]
with ftputil.FTPHost(server,login,pwd) as host: # ftp host info
    recursive = host.walk("/",topdown=True,onerror=None) # recursive search
    pp = 0
    for root,dirs,files in recursive:
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                rfile = host.path.join(root, name)
                lfile = os.path.join(cfx_path, name)
                p = host.download_if_newer(rfile, lfile)
                if p: print "Downloading file %s" % rfile
                pp = pp + int(p)
    print "Downloaded %d files" % pp

host.close()