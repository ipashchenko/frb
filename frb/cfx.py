# -*- coding: utf-8 -*-
""" Dealing with cfx files """

import os, re, fnmatch
#from config import config

# TODO: move this func to tools/utils module
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
    """
    Get latest version cfx file(s) for given exp code.
    Return list of filenames
    """
    cfxpattern = 'radioastron_%s*.cfx' % code.lower()
    cfxlist = []
    last_ver = []
    cfx_bands = set()
    for fname in os.listdir(cfx_path):
        if fnmatch.fnmatch(fname.lower(), cfxpattern):
            cfx = CFX(fname)
            cfx_bands.add(cfx.band())
            cfxlist.append(fname)
    for band in cfx_bands:
        band_list = fnmatch.filter(cfxlist, '*_{}_*'.format(band))
        last_ver.append(os.path.join(cfx_path,
        sorted(band_list, key=lambda x: CFX(x).version())[-1]))
    return last_ver

class CFX(object):
    def __init__(self, cfile):
        self.cfile = cfile
        self.fname = os.path.basename(self.cfile)
    def version(self):
        """get CFX version from file name"""
        a = re.search('(?<=_V)\d{1,2}', self.fname)
        if a is None:
            return None
        else:
            return int(a.group())
    def band(self):
        """get CFX band (K, C, L, P) from file name"""
        a = re.search('(?<=_)[K,C,L,P]{1}(?<!_)', self.fname)
        if a is None:
            return None
        else:
            return a.group()

    def parse_cfx(self, code=""):
        """ CFX files parsing. Return dictionary {Data-file: parameters} """
        cfxdata = dict()
#       code = ra_code(cfxfile)
        f = open(self.cfile)
        txt = f.read()
        f.close()
        reg = "\[\$end\](?i)"
        blocks = re.split(reg, txt)  # blocks in file
        for ind, block in enumerate(blocks):
            if '[$TLSC]' in block:
    # TODO: deal with spaces (???)
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

#    def process(self):
#        """ some manipulations: skip K-band, link files, ..."""
#        cdat = self.parse_cfx(self, code)
#        if self.band() == 'K':
#            print("Skipping K-band CFX file: {}".format(os.path.basename(cfile)))
#            print("NOTE: You can delete following files from data path:\n")
#            print cfx_data.keys()
#            return
1



if __name__ == "__main__":
    cfx_path = '/home/osh/frb_test/cfx'
    code = 'raes03gb'
    cfiles = get_cfx(cfx_path, code)
    c = CFX(cfiles[1]).parse_cfx(code)
    print c
