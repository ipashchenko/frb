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

# Default SEFDs of antenna
sefd_dict = {'RADIO-AS': {'K': {'L': 46700., 'R': 36800},
                          'C': {'L': 11600., 'R': None},
                          'L': {'L': 2760., 'R': 2930.}},
             'GBT-VLBA': {'K': {'L': 23., 'R': 23.},
                          'C': {'L': 8., 'R': 8.},
                          'L': {'L': 10., 'R': 10.}},
             'EFLSBERG': {'C': {'L': 20., 'R': 20.},
                          'L': {'L': 19., 'R': 19.}},
             'YEBES40M': {'C': {'L': 160., 'R': 160.},
                          'L': {'L': None, 'R': None}},
             'ZELENCHK': {'C': {'L': 400., 'R': 400.},
                          'L': {'L': 300., 'R': 300.}},
             'EVPTRIYA': {'C': {'L': 44., 'R': 44.},
                          'L': {'L': 44., 'R': 44.}},
             'SVETLOE': {'C': {'L': 250., 'R': 250.},
                         'L': {'L': 360., 'R': 360.}},
             'BADARY': {'C': {'L': 200., 'R': 200.},
                        'L': {'L': 330., 'R': 330.}},
             'TORUN': {'C': {'L': 220., 'R': 220.},
                       'L': {'L': 300., 'R': 300.}},
             'ARECIBO': {'C': {'L': 5., 'R': 5.},
                         'L': {'L': 3., 'R': 3.}},
             'WSTRB-07': {'C': {'L': 120., 'R': 120.},
                          'L': {'L': 40., 'R': 40.}},
             'VLA-N8': {'C': {'L': None, 'R': None},
                        'L': {'L': None, 'R': None}},
             # Default values for KL
             'KALYAZIN': {'C': {'L': 150., 'R': 150.},
                          'L': {'L': 140., 'R': 140.}},
             'MEDICINA': {'C': {'L': 170., 'R': 170.},
                          'L': {'L': 700., 'R': 700.}},
             'NOTO': {'C': {'L': 260., 'R': 260.},
                      'L': {'L': 784., 'R': 784.}},
             'HARTRAO': {'C': {'L': 650., 'R': 650.},
                         'L': {'L': 430., 'R': 430.}},
             'HOBART26': {'C': {'L': 640., 'R': 640.},
                          'L': {'L': 470., 'R': 470.}},
             'MOPRA': {'C': {'L': 350., 'R': 350.},
                       'L': {'L': 340., 'R': 340.},
                       'K': {'L': 900., 'R': 900.}},
             'WARK12M': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'TIDBIN64': {'C': {'L': None, 'R': None},
                          'L': {'L': None, 'R': None}},
             'DSS63': {'C': {'L': 24., 'R': 24.},
                       'L': {'L': 24., 'R': 24.}},
             'PARKES': {'C': {'L': 110., 'R': 110.},
                        'L': {'L': 40., 'R': 40.},
                        'K': {'L': 810., 'R': 810.}},
             'USUDA64': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}},
             'JODRELL2': {'C': {'L': 320., 'R': 320.},
                          'L': {'L': 320., 'R': 320.}},
             'ATCA104': {'C': {'L': None, 'R': None},
                         'L': {'L': None, 'R': None}}}
                         
# TODO: Implement ``band``, ``ignored`` & ``loss_func`` functionality                         
def select_antenna(antenna, band, n_small=2, n_big=1, d_lim=50., ignored=None,
                   loss_func=None):
    """
    Function that selects antenna.
    
    :param antenna:
        Iterable of antenna.
    :param band:
        Frequency band. (``K``, ``C``, ``L`` or ``P``)
    :param n_small: (optional)
        Number of small antenn to select. (default: ``2``)
    :param n_big: (optional)
        Number of big antenna to select. (default: ``1``)
    :param d_lim: (optional)
        Diameter that defines ``big`` and ``small`` antenna. (default: ``50.0``)
    :param ignored: (optional)
        Iterable of ignored antenna. If ``None`` then don't ignore any.
        (default: ``None``)
    :param loss_func: (optional)
        Callable that represents a loss function assosciated with given diameter ``d``
        and SEFD ``sefd``. ``loss_func(d, sefd)`` represents losses. If ``None`` then
        don't use it in selecting antenna.
        
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
    
      
    
