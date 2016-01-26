# -*- coding: utf-8 -*-
""" Dealing with cfx files """

import os, re, fnmatch
#from config import config

def ra_code(string):
    """ find RA code in string """
    code_pattern = 'ra{0,1}[efgk]s{0,1}\d{2}[a-z][0-9a-z]{0,1}'
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

def get_cfx(cfx_path, code):
    """ """
    cfxpattern = 'radioastron_%s*.cfx' % code.lower()
    cfx_files = dict()
    cfxlist = []
    for fname in os.listdir(cfx_path):
        if fnmatch.fnmatch(fname.lower(), cfxpattern):
            cfx_version = re.search('(?<=_V)\d{1,2}', fname)
            cfx_band = re.search('(?<=_)[K,C,L,P]{1}(?<!_)', fname)
            cfx_files.update({fname : [cfx_version.group(), cfx_band.group()]})
            cfxlist.append(fname)
    if not cfxlist:
        print "No CFX files found for %s" % code
        return
    return sorted(cfx_files.items(), key=lambda x:int(x[1][0]))[::-1]


def parse_cfx(cfxfile):
    """ CFX files parsing. Return dictionary {Data-file: parameters} """
    cfxdata = dict()
#    code = ra_code(cfxfile)
    f = open(cfxfile)
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


if __name__ == "__main__":
    cfx_path = '/home/osh/frb_test/cfx'
    code = 'raes03gb'
    a = get_cfx(cfx_path, code)
    print a
#    cfile = "/home/osh/Dropbox/frb/data/GLVBI_RAKS01GR_C_20131022T005200_ASC_V1.cfx"
#    c = parse_cfx(cfx_path + '/' + a[1])
#    print c
