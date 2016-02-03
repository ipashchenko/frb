import os
import glob
import astropy.io.fits as pf
from fits_io import get_dyn_spectr
from antenna import select_antenna
from dedispersion import de_disperse


RAW_DIR = '/mnt/frb_data/raw_data/'


def process_experiment(exp_name, t_size=600000):
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
            process_dyn_spectra(t, nu, dsp_i)


# de-dispersion, finding candidates, saving them
def process_dyn_spectra(t, nu, dsp):
    pass

