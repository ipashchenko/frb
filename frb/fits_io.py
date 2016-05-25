import numpy as np
import astropy.io.fits as pf
from astropy.time import Time


def find_card_from_header(header, value=None, keyword=None,
                          comment_contens=None):
    """
    Find card from header specified be several possible ways.

    :param header:
        Instance of ``astropy.io.fits.Header`` class.
    :param value:
        Value of header's card that specifies card.
    :param keyword:
        Keyword of header's card that specifies card.
    :param comment_contens:
        Comment of header's card that specifies card.

    :return:
        Instance of ``astropy.io.fits.card.Card`` class.
    """
    if comment_contens is not None:
        search = [card for card in header.cards if comment_contens in
                  card.comment]
    else:
        search = header.cards

    if value is not None and keyword is None:
        result = [card for card in search if card.value == value]
    elif value is None and keyword is not None:
        result = [card for card in search if card.keyword == keyword]
    elif value is not None and keyword is not None:
        result = [card for card in search if (card.keyword == keyword and
                                              card.value == value)]
    else:
        result = search

    return result


def get_key(header, value, keyword):
    """
    Get some keyword value from header.

    :param header:
        Instance of ``astropy.io.fits.Header`` class.
    :param value:
        Value of header's card that specifies parameter.
    :param keyword:
        Key to value to return.

    :return:
        Value for specified key.
    """
    freq_card = find_card_from_header(header, value=value)[0]
    return header[keyword + '{}'.format(freq_card[0][-1])]


def get_dyn_spectr(fits_idi, band=None, channel=None, time=None,
                   complex_indx=None, stokes_indx=None):
    """
    Function that reads specified part of `regular` data matrix and returns
    it with time and frequency information.

    :param fits_idi:
        Path to FITS file.
    :param band: (optional)
        Slice that defines band numbers. If ``None`` then concatenate all.
        (default: ``None``)
    :param channel: (optional)
        Slice that defines channel numbers. If ``None`` then output all.
        (default: ``None``)
    :param time: (optional)
        Slice that defines times. If ``None`` then use sensible default.
        (default: ``None``)
    :param complex_indx: (optional)
        Index of COMPLEX regular data matrix to output. If ``None`` then output
        all. (default: ``None``)
    :param stokes_indx: (optional)
        Index of STOKES regular data matrix to output. If ``None`` then output
        all. (default: ``None``)

    :return:
        nu, t & Numpy 2D array of (#nu, #t), where #nu - number of frequency
        channels and #t - number of time intervals. ``nu`` - frequencies [Hz]
        and ``t`` - times.
    """
    hdulist = pf.open(fits_idi)

    try:
        indx = hdulist.index_of('UV_DATA')
        hdu = hdulist[indx]
    except KeyError:
        import traceback
        print("No UV_DATA extension in {}".format(fits_idi))
        print("  exception:")
        traceback.print_exc()
        raise
    try:
        indx = hdulist.index_of('FREQUENCY')
        fhdu = hdulist[indx]
    except KeyError:
        import traceback
        print("No FREQUENCY extension in {}".format(fits_idi))
        print("  exception:")
        traceback.print_exc()
        raise

    n_band = hdu.header['NO_BAND']
    n_chan = hdu.header['NO_CHAN']
    n_stok = hdu.header['NO_STKD']
    ref_freq = hdu.header['REF_FREQ']
    n_cmplx = get_key(hdu.header, 'COMPLEX', 'MAXIS')

    if complex_indx is None:
        complex_indx = slice(0, n_cmplx)
    if stokes_indx is None:
        stokes_indx = slice(0, n_stok)
    if channel is None:
        channel = slice(0, n_chan)
    if band is None:
        band = slice(0, n_band)

    times = Time(hdu.data['DATE'][time] +
                 hdu.data['TIME'][time], format='jd')

    crpix = get_key(hdu.header, 'FREQ', 'CRPIX')

    channels = np.arange(n_chan) + 1
    # FIXME: let FREQID be ``1`` everywhere
    # fhdu_id = fhdu.data[np.where(fhdu.data['FREQID'] == freq_id)]
    # (#band, #channels) array of frequencies
    frequencies = (ref_freq + fhdu.data['BANDFREQ'][..., band] +
                   (channels[channel] - crpix)[:, np.newaxis] *
                   fhdu.data['CH_WIDTH'][..., band]).T

    data = hdu.data['FLUX'][time, ...]
    data = np.reshape(data, (data.shape[0], n_band, n_chan, n_stok, n_cmplx))

    if len(range(n_band)[band]) > 1:
        result = list()
        for i in range(n_band)[band]:
            result.append(data[:, i, channel, stokes_indx, complex_indx])
        result = np.concatenate(result, axis=1)
    else:
        result = data[:, band, channel, stokes_indx, complex_indx]

    return times, frequencies, result.T


if __name__ == '__main__':
    idi_fits = '/mnt/frb_data/raw_data/re03jy/RE03JY_EF_C_AUTO.idifits'
    t, nu_array, dsp = get_dyn_spectr(idi_fits, time=slice(0, 100000),
                                      complex_indx=0, stokes_indx=0)
    dsp += get_dyn_spectr(idi_fits, time=slice(0, 100000), complex_indx=0,
                          stokes_indx=1)[2]
    dsp *= 0.5
    nu_max = np.max(nu_array.ravel()) / 10 ** 6
    d_nu = (nu_array[0][1:] - nu_array[0][:-1])[0] / 10 ** 6
    d_t = (t[1] - t[0]).sec
