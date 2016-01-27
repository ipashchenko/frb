# Diameters
diameters_dict = {'RADIO-AS': 32., 'GBT-VLBA': 30., 'EFLSBERG': 100.,
                  'YEBES40M': 40., 'ZELENCHK': 32., 'EVPTRIYA': 70.,
                  'SVETLOE': 32., 'BADARY': 32., 'TORUN': 32.,
                  'ARECIBO': 300., 'WSTRB-07': 27.,  'VLA-N8': 30.,
                  'KALYAZIN': 64., 'MEDICINA': 30., 'NOTO': 30.,
                  'HARTRAO': 20., 'HOBART26': 26., 'MOPRA': 30.,
                  'WARK12M': 12., 'TIDBIN64': 64., 'DSS63': 63.,
                  'PARKES': 64., 'USUDA64': 64., 'JODRELL2': 64.,
                  'ATCA104': 104.}


def select_antenna(antenna, n_small=2, n_big=1, d_lim=50., ignored=None):
    """
    Function that selects antenna.

    :param antenna:
        Iterable of antenna.
    :param n_small: (optional)
        Number of small antenna to select. (default: ``2``)
    :param n_big: (optional)
        Number of big antenna to select. (default: ``1``)
    :param d_lim: (optional)
        Diameter that defines ``big`` and ``small`` antenna. (default: ``50.0``)
    :param ignored: (optional)
        Iterable of ignored antenna. If ``None`` then don't ignore any.
        (default: ``None``)

    :return:
        List of antenna. First goes ``n_big`` big and remaining are ``n_small``
        small.
    """
    antenna = list(antenna)[:]
    if ignored:
        for ant in ignored:
            antenna.remove(ant)
    antenna = sorted(antenna, key=lambda x: diameters_dict[x])
    try:
        big_list = [ant for ant in antenna if diameters_dict[ant] > d_lim]
    except KeyError:
        print("Provide diameter value for antenna {}".format(ant))
    # Check - at least ``n_big`` big antenna in list
    if len(big_list) < n_big:
        raise Exception("No {} telescopes with D > {} m in list".format(n_big, d_lim))
    small_list = [ant for ant in antenna if diameters_dict[ant] < d_lim]
    # Check - at least ``n_small`` small antenna in list
    if len(small_list) < n_small:
        raise Exception("No {} telescopes with D < {} m in list".format(n_small, d_lim))

    return big_list[: n_big], small_list[: n_small]


if __name__ == '__main__':
    antenna = ['RADIO-AS', 'ZELENCHK', 'NOTO', 'USUDA64', 'SVETLOE', 'ARECIBO']
    selected = select_antenna(antenna)
    print(selected)
    selected = select_antenna(antenna, ignored=['RADIO-AS'])
    print(selected)
    selected = select_antenna(antenna, n_big=2, n_small=3, ignored=['RADIO-AS'])
    print(selected)

