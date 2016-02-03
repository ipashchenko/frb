import os
import glob
import astropy.io.fits as pf
import numpy as np
from fits_io import get_dyn_spectr
from dedispersion import de_disperse
from search import search_candidates


RAW_DIR = '/mnt/frb_data/raw_data/'


def process_experiment(exp_name, dm_values, t_size=600000, ddsp_kwargs=None,
                       search_kwargs=None):
    exp_dir = os.path.join(RAW_DIR, exp_name)
    fits_files = glob.glob(os.path.join(exp_dir, "*fits"))

    if ddsp_kwargs is None:
        ddsp_kwargs = dict()

    # Create mapping `antenna - fits-file`
    ant_file_map = dict()
    for fname in fits_files:
        ant = set(pf.getdata(fname, extname='ANTENNA')['ANNAME'])
        assert len(ant) == 1
        ant = ant.pop()
        ant_file_map.update({ant: fname})

    # Find what fits files we need to process
    availale_ant = set(ant_file_map.keys())
    for ant in availale_ant:
        print("Working with antenna {}".format(ant))
        ant_fname = ant_file_map[ant]
        print("Antenna file: {}".format(ant_fname))
        n_t = pf.getval(ant_fname, 'NAXIS2', extname='UV_DATA')
        n = n_t // t_size
        t_slices = [(i * t_size, (i + 1) * t_size) for i in range(n)]
        t_slices += [(n * t_size, n_t)]
        t_slices = [slice(*t_slice) for t_slice in t_slices]
        print("Slices: {}".format(t_slices))
        for t_slice in t_slices:
            # First get `RR`
            t, nu_array, dsp = get_dyn_spectr(ant_fname, complex_indx=0,
                                              stokes_indx=0, time=t_slice)
            # Then add `LL`
            dsp += get_dyn_spectr(ant_fname, complex_indx=0, stokes_indx=1,
                                  time=t_slice)[2]
            # Finally, calculate `I`
            dsp *= 0.5
            print("Will work with time interval:")
            print("From {} to {}".format(t[0].utc.iso, t[-1].utc.iso))
            # De-dispersion kwargs
            nu_max = np.max(nu_array.ravel()) / 10 ** 6
            d_nu = (nu_array[0][1:] - nu_array[0][:-1])[0] / 10 ** 6
            d_t = (t[1] - t[0]).sec
            ddsp_kwargs.update({'nu_max': nu_max, 'd_nu': d_nu, 'd_t': d_t})
            candidates = process_dyn_spectra(t, dsp, dm_values,
                                             ddsp_kwargs=ddsp_kwargs,
                                             search_kwargs=search_kwargs)
            print(candidates)


# de-dispersion, finding candidates, saving them to DB
def process_dyn_spectra(t, dsp, dm_values,  ddsp_kwargs=None,
                        search_kwargs=None):
    # Get de-dispersion parameters and de-disperse
    print("De-dispersing...")
    tdm = de_disperse(dsp, dm_values, **ddsp_kwargs)
    # Get search candidates parameters and serach for candidates
    print("Searching candidates...")
    candidates = search_candidates(tdm, **search_kwargs)
    return candidates


if __name__ == '__main__':
    exp_name = 'RE03JY'
    dm_values = np.arange(0., 1500., 50.)
    search_kwargs = {'threshold': 99.55, 'n_d_x': 3., 'n_d_y': 5.}

    process_experiment(exp_name.lower(), dm_values, t_size=300000,
                       search_kwargs=search_kwargs)


