# -*- coding: utf-8 -*-
""" Dealing with cfx files """

import os
import re
import fnmatch
# from config import config


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
            cfx_bands.add(cfx.freq)
            cfxlist.append(fname)
    for band in cfx_bands:
        band_list = fnmatch.filter(cfxlist, '*_{}_*'.format(band))
        last_ver.append(os.path.join(cfx_path,
                                     sorted(band_list,
                                            key=lambda x:
                                            CFX(x).version())[-1]))
    return last_ver


# TODO: Check if file exists in constructor
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

    @property
    def freq(self):
        """get CFX freq (K, C, L, P) from file name"""
        a = re.search('(?<=_)[K,C,L,P]{1}(?<!_)', self.fname)
        if a is None:
            raise Exception("Can't determine frequency freq from CFX file")
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
                # add full telescope name
                # WTF with "code : here ???
                tname = re.search('(?<=iam_name = )[A-Za-z]+', block)
                fmt = re.search('(?<=FORMAT.. = )[A-Za-z0-9_\-]+', block)
                ifs = re.findall('(?<=IF = )([0-9\.]+\, [RL]\, [UL])', block)
                files = re.findall('(?<=FILE.. = %P:)([\S]*)', block)
                print "ifs", ifs
                cfx_ifs = ' '.join(ifs)
                cfx_fmt = re.sub('\, ', '-', cfx_ifs)
                val1 = {'exp_code': code, 'antenna': tname.group(0),
                        'm5_fmt': fmt.group(0), 'cfx_fmt': cfx_fmt.split(),
                        'freq': self.freq.lower(),
                        'band': [if_.split()[2] for if_ in ifs],
                        'pol': [if_.split()[1].strip(',') for if_ in ifs]}
                for f in files:
                    cfxdata.update({f: val1})
        return cfxdata
