# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:23:56 2016

@author: osh
"""
### define global Dropbox path and user scrips path
import sys, os

def dir_scan(datadir):
    """ Scan a directory and output a list of files """
    datadir = os.path.abspath(datadir)
    filelist = []
    for root, subdirs, files in os.walk(datadir):
        for filename in files:
            file_path = os.path.join(root, filename)
            filelist.append(file_path)
    return filelist



