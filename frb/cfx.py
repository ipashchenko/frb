# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:10:44 2016

@author: osh
"""
### define global Dropbox path and user scrips path
import re

def dir_scan(datadir):
    datadir = os.path.abspath(datadir)
    filelist = []
    for root, subdirs, files in os.walk(datadir):
        for filename in files:
            file_path = os.path.join(root, filename)
            filelist.append(file_path)
    return filelist

def ra_code(string):
    """ find RA code in string """
    code = re.search(code_pattern, string.lower())
    if not code:
        print "No code found"
        return
    c = code.group()
    if c[:1] == 'rk':
        code = 'raks' + c[2:]
    elif c[:1] == 're':
        code = 'raes' + c[2:]
    elif c[:1] == 'rg':
        code = 'rags' + c[2:]
    elif c[:1] == 'rf':
        code = 'rafs' + c[2:]
    else:
        code = c
    return code

def get_cfx(code):

    cfxpattern = 'radioastron_%s*.cfx' % code.lower()
    cfxlist = []
    for fname in os.listdir(cfx_dir):
        if fnmatch.fnmatch(fname.lower(), cfxpattern):
            cfxlist.append(fname)
    if not cfxlist:
        print "No CFX files found for %s" % code
        return
# TODO: cfx-versions
#        fvers=[]
#        for fname in self.cfxlist:
#            fver = np.int(fname[-5])
#            if fver == 0: fver=10
#            fvers.append(fver)
#        m = max(fvers)
#        print m
#            print fname, fver, datetime.fromtimestamp(os.path.getmtime(self.cfx_dir + fname))
    return cfxlist


def parse_cfx(cfxfiles):
    """ CFX files parsing. Return dictionary {CFX-file: parameters} """
    if not cfxfiles:
        print "No cfx files"
        return
    if isinstance(cfxfiles, str): cfxfiles = list(cfxfiles)
    cfxdata = dict()
    for cfxfile in cfxfiles:
        code = ra_code(cfxfile)
        f = open(cfx_dir + cfxfile)
        txt = f.read()
        f.close()
        reg = "\[\$end\](?i)"
        blocks = re.split(reg, txt)  # blocks in file
        for ind, block in enumerate(blocks):
            if '[$TLSC]' in block:
# TODO: deal with spaces
                tname = re.search('(?<=iam_name = )[A-Za-z]+', block)
                fmt = re.search('(?<=FORMAT.. = )[A-Za-z0-9_\-]+', block)
                ifs = re.findall('(?<=IF = )([0-9\.]+\, [RL]\, [UL])', block)
                files = re.findall('(?<=FILE.. = %P:)([\S]*)', block)
                ifs = ' '.join(ifs)
                ifs = re.sub('\, ', '-', ifs)
                val1 = [code, tname.group(0), fmt.group(0), ifs.split()]
                for f in files:
                    cfxdata.update({f:val1})
    return cfxdata



