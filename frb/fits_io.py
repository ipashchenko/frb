import re
import string
import numpy as np
import astropy.io.fits as pf


def aips_bintable_fortran_fields_to_dtype_conversion(aips_type):
    """
    Given AIPS fortran format of binary table (BT) fields, returns
    corresponding numpy dtype format and shape.

    :param aips_type:
        String of AIPS type.

    :examples:
        4J => array of 4 32bit integers,
        E(4,32) => two dimensional array with 4 columns and 32 rows.
    """

    intv = np.vectorize(int)
    aips_char = None

    format_dict = {'L': 'bool', 'I': '>i2', 'J': '>i4', 'A': 'S',  'E': '>f4',
                   'D': '>f8'}

    for key in format_dict.keys():
        if key in aips_type:
            aips_char = key

    if not aips_char:
        raise Exception("aips data format reading problem " + str(aips_type))

    try:
        dtype_char = format_dict[aips_char]
    except KeyError:
        raise Exception("no dtype counterpart for aips data format" +
                        str(aips_char))

    try:
        repeat = int(re.search(r"^(\d+)" + aips_char,
                     aips_type).groups()[0])
        if aips_char is 'A':
            dtype_char = str(repeat) + dtype_char
            repeat = 1
    except AttributeError:
        repeat = None

    if repeat is None:
        _shape = tuple(intv(string.split(re.search(r"^" + aips_char +
                                                   "\((.+)\)$",
                                                   aips_type).groups()[0],
                                         sep=',')))
    else:
        _shape = repeat

    return dtype_char, _shape


def build_dtype_for_bintable_data(header):
    """
    Builds numpy dtype from instance of ``astropy.HDU`` class.

    :return:
        List with dtype specification and list of regular parameter names.

    """

    # Number of fields in a item
    tfields = int(header['TFIELDS'])

    # Parameters of regular data matrix if in UV_DATA FITS-IDI table
    try:
        maxis = int(header['MAXIS'])
    except KeyError:
        import traceback
        print("No UV_DATA extension")
        print("  exception:")
        traceback.print_exc()
        raise

    # build dtype format
    names = []
    formats = []
    shapes = []
    tuple_shape = []
    array_names = []

    for i in range(1, tfields + 1):
        name = header['TTYPE' + str(i)]
        if name in names:
            name *= 2
        names.append(name)
        _format, _shape = \
            aips_bintable_fortran_fields_to_dtype_conversion(header['TFORM' +
                                                                    str(i)])

        # building format & names for regular data matrix
        if name == 'FLUX' and maxis is not None:
            for i in range(1, maxis + 1):
                maxisi = int(header['MAXIS' + str(i)])
                if maxisi > 1:
                    tuple_shape.append(int(header['MAXIS' + str(i)]))
                    array_names.append(header['CTYPE' + str(i)])
            formats.append('>f4')
            shapes.append(tuple(tuple_shape))
            array_names = array_names
        else:
            formats.append(_format)
            shapes.append(_shape)

    print names, formats, shapes, array_names

    dtype_builder = zip(names, formats, shapes)
    dtype = [(name, _format, shape) for (name, _format, shape) in dtype_builder]

    return dtype, array_names


def fits_idi_uv_data_info(fname):
    hdulist = pf.open(fname)

    try:
        indx = hdulist.index_of('UV_DATA')
        hdu = hdulist[indx]
    except KeyError:
        import traceback
        print("No UV_DATA extension in {}".format(fname))
        print("  exception:")
        traceback.print_exc()
        raise
    dtype, regular_names = build_dtype_for_bintable_data(hdu.header)
    dtype_names = [d[0] for d in dtype]
    try:
        idx = dtype_names.index('FLUX')
    except ValueError:
        import traceback
        print("No FLUX field in regular data")
        print("  exception:")
        traceback.print_exc()
        raise
    flux_dtype = dtype[idx]
    print("{} measurements".format(hdu.header['NAXIS2']))
    for name, shape in zip(regular_names, flux_dtype[-1]):
        print("{} dimensions for regular parameter {}".format(shape, name))

