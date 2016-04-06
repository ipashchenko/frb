# -*- coding: utf-8 -*-
"""
Monotoring of directory for new files
Created on Mon Mar 28 17:39:24 2016

@author: osh
"""
import os

def dict_compare(d1, d2):
    """ from http://stackoverflow.com/questions/4527942"""
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same

def is_indexed(path):
    """ Check if path is being indexed """
    indfile = path + '/.index'
    if not os.path.exists(indfile) or os.stat(indfile).st_size == 0:
        return False
    else:
        return True

def read_index(path):
    """ reading index file in given path to dictionary"""
    dic = dict()
    indfile = path + '/.index'
    if is_indexed(path):
        f = open(indfile)
        for line in f.readlines():
            x = line.split()
            k = ' '.join(y for y in x[:-1])
            v = x[-1]
            dic.update({k:int(v)})
        f.close()
        return dic
    else:
        return None

def write_index(path, dic):
    """ write index file from given dict"""
    fid = open(path + "/.index", 'w')

    for k, v in dic.items():
        fid.write("{:70} {}\n".format(k,v))
    fid.close()

def read_path(path):
    """ reading all files|dirs in path returning dict"""
    dic = dict()
    for dirpath, dirname, filenames in os.walk(path):
        for filename in filenames:
            fname = os.path.join(dirpath, filename)
            mtime = int(os.stat(fname).st_mtime)
            dic.update({fname: mtime})
    return dic

def anynews(path):
    old = read_index(path)
    new = read_path(path)
#    print new
    if old is None:
        write_index(path, new)
        print("Creating new index file")
        return
    added, removed, modified, same = dict_compare(new, old)
    added.discard(path + '/.index')  # skip index file
    if path + '/.index' in modified: modified.pop(path + '/.index')
    print "added:", added
    print "removed:", removed
    print "modified:", modified
    if added or removed or modified:
        write_index(path, new)
        print "Updating index"
    return 0

#    print same


if __name__ == "__main__":
    path = '/home/osh/frb_test/raw_data'
#    anynews(path)
#    print read_index(path)
#    a = {1:'adfa', 2:'qedfdcad', 'm':'afadfadfa'}
#    b = a.items()
#    print b

#    x = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
#    y = {'Name': 'Zara', 'Age': 8, 'Class': 'First'}
#
#    added, removed, modified, same = dict_compare(x, y)
#    print added
#    print removed
#    print modified
#    print same
    anynews(path)