# Diameters
diameters_dict = {'RADIO-AS': None, 'GBT-VLBA': None, 'EFLSBERG': None,
                  'YEBES40M': None, 'ZELENCHK': None, 'EVPTRIYA': None,
                  'SVETLOE': None, 'BADARY': None, 'TORUN': None,
                  'ARECIBO': None, 'WSTRB-07': None,  'VLA-N8': None,
                  'KALYAZIN': None, 'MEDICINA': None, 'NOTO': None,
                  'HARTRAO': None, 'HOBART26': None, 'MOPRA': None,
                  'WARK12M': None, 'TIDBIN64': None, 'DSS63': None,
                  'PARKES': None, 'USUDA64': None, 'JODRELL2': None,
                  'ATCA104': None}

                         
def select_antenna(antenna, n_small=2, n_big=1, d_lim=50., ignored=None):
    """
    Function that selects antenna.
    
    :param antenna:
        Iterable of antenna.
    :param n_small: (optional)
        Number of small antenn to select. (default: ``2``)
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
    big_list = list()
    small_list = list()
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
    
    return big_list, small_list    
    
      
    
