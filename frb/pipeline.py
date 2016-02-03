import os
import glob
import astropy.io.fits as pf
import numpy as np
from fits_io import get_dyn_spectr
from antenna import select_antenna
from dedispersion import de_disperse
from objects import search_candidates


RAW_DIR = '/mnt/frb_data/raw_data/'

ddsp_kwargs = {'nu_max': None, 'd_nu': None, 'd_t': None}
search_kwargs = {'threshold': 99.85, 'd_dm': 300., 'd_t': 0.003,
                 'batch_size': 100000}


def process_experiment(exp_name, dm_values, t_size=600000, ddsp_kwargs=None,
                       search_kwargs=None):
    exp_dir = os.path.join(RAW_DIR, exp_name)
    fits_files = glob.glob(os.path.join(exp_dir, "*fits"))

    # Create mapping `antenna - fits-file`
    ant_file_map = dict()
    for fname in fits_files:
        ant = set(pf.getdata(fname, extname='ANTENNA')['ANNAME'])
        assert len(ant) == 1
        ant = ant.pop()
        ant_file_map.update({ant: fname})

    # Find what fits files we need to process
    availale_ant = set(ant_file_map.keys())
    big_ant, small_ant = select_antenna(availale_ant, n_small=2, n_big=1,
                                        d_lim=50., ignored=None)
    antenna = big_ant + small_ant
    for ant in antenna:
        ant_fname = ant_file_map[ant]
        n_t = pf.getval(ant_fname, 'NAXIS2', extname='UV_DATA')
        n = n_t // t_size
        t_slices = [(i * t_size, (i + 1) * t_size) for i in range(n)]
        t_slices += [(n * t_size, n_t)]
        for t_slice in t_slices:
            t, nu, dsp_rr = get_dyn_spectr(ant_fname, complex_indx=0,
                                           stokes_indx=0, time=t_slice)
            t, nu, dsp_ll = get_dyn_spectr(ant_fname, complex_indx=0,
                                           stokes_indx=1, time=t_slice)
            dsp_i = 0.5 * (dsp_ll + dsp_rr)
            # De-dispersion kwargs
            nu_max = np.max(nu.ravel()) / 10 ** 6
            d_nu = (nu[0][1:] - nu[0][:-1])[0] / 10 ** 6
            d_t = (t[1] - t[0]).sec
            ddsp_kwargs.update({'nu_max': nu_max, 'd_nu': d_nu, 'd_t': d_t})
            process_dyn_spectra(t, nu, dsp_i, ddsp_kwargs=ddsp_kwargs,
                                search_kwargs=search_kwargs)


# de-dispersion, finding candidates, saving them to DB
def process_dyn_spectra(t, dsp, dm_values,  ddsp_kwargs=None,
                        search_kwargs=None):
    # Get de-dispersion parameters and de-disperse
    tdm = de_disperse(dsp, dm_values, **ddsp_kwargs)
    # Get search candidates parameters and serach for candidates
    candidates = search_candidates(tdm, t, dm_values, **search_kwargs)
    return candidates

