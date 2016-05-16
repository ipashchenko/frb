import numpy as np
from astropy.time import Time
from frb.frames import create_from_txt
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import (search_candidates, search_candidates_ell,
                        create_ellipses)


print "Creating Dynamical Spectra"
# frame = Frame(256, 10000, 1684., 0., 16./256, 1./1000)
txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
frame = create_from_txt(txt, 1684., 0, 16./128, 0.001)
t0 = Time.now()
print "Start time {}".format(t0)
# Number of artificially injected pulses
n_pulses = 30
# Step of de-dispersion
d_dm = 30.
print "Adding {} pulses".format(n_pulses)

# Set random generator seed for reproducibility
np.random.seed(123)
# Generate values of pulse parameters
amps = np.random.uniform(0.25, 0.35, size=n_pulses)
widths = np.random.uniform(0.001, 0.003, size=n_pulses)
dm_values = np.random.uniform(100, 500, size=n_pulses)
times = np.linspace(0.1, frame.shape[0] - 0.1, n_pulses)
# Injecting pulses
for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
    frame.add_pulse(t_0, amp, width, dm)
    print "Adding pulse with t0={}, amp={}, width={}, dm={}".format(t_0, amp,
                                                                    width, dm)

meta_data = {'antenna': 'WB', 'freq': 'L', 'band': 'U', 'pol': 'R',
             'exp_code': 'raks00', 'nu_max': 1684., 't_0': t0, 'd_nu': 16./128.,
             'd_t': 0.001}
# Values of DM to de-disperse
dm_grid = np.arange(0., 1000., d_dm)
# Initialize searcher class
searcher = Searcher(dsp=frame.values, meta_data=meta_data)
# Run search for FRB with some parameters of de-dispersion, pre-processing,
# searching algorithms
print "using ``search_candidates`` search function..."
candidates = searcher.run(de_disp_func=de_disperse_cumsum,
                          search_func=search_candidates,
                          preprocess_func=create_ellipses,
                          de_disp_args=[dm_grid],
                          search_kwargs={'n_d_x': 4., 'n_d_y': 15.,
                                         'd_dm': d_dm},
                          preprocess_kwargs={'disk_size': 3,
                                             'threshold_big_perc': 97.5,
                                             'threshold_perc': 98.5,
                                             'statistic': 'mean'})
print "Found {} candidates".format(len(candidates))
for candidate in candidates:
    print candidate

# Run search for FRB with same parameters of de-dispersion, but different
# pre-processing & searching algorithms
print "using ``search_candidates_ell`` search function..."
candidates = searcher.run(de_disp_func=de_disperse_cumsum,
                          search_func=search_candidates_ell,
                          preprocess_func=create_ellipses,
                          de_disp_args=[dm_grid],
                          search_kwargs={'x_stddev': 6., 'y_to_x_stddev': 0.3,
                                         'theta_lims': [130., 180.],
                                         'x_cos_theta': 3.,
                                         'd_dm': d_dm,
                                         'amplitude': 3.},
                          preprocess_kwargs={'disk_size': 3,
                                             'threshold_big_perc': 97.5,
                                             'threshold_perc': 98.5,
                                             'statistic': 'mean'})
print "Found {} candidates".format(len(candidates))
for candidate in candidates:
    print candidate

# # Now change parameters of just search phase - using calculated
# # ``Searcher._de_dispersed_data`` & ``_preprocessed_data``
# # Candidates & searched data won't go to DB when calling ``Searcher.search``
# # explicitly!
# candidates = searcher.search(search_candidates, n_d_x=8., n_d_y=15.,
#                              d_dm=d_dm)
# print "Found {} pulses".format(len(candidates))
# for candidate in candidates:
#     print candidate
#
# # Going through all pipeline & using cached de-dispersed values because
# # preprocessing parameters have changed
# candidates = searcher.run(de_disp_func=de_disperse_cumsum,
#                           search_func=search_candidates,
#                           preprocess_func=create_ellipses,
#                           de_disp_args=[dm_grid],
#                           de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256},
#                           search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
#                                          'd_dm': d_dm},
#                           preprocess_kwargs={'disk_size': 3,
#                                              'threshold_perc': 95.,
#                                              'statistic': 'mean'})
# print "Found {} pulses".format(len(candidates))
# for candidate in candidates:
#     print candidate
#
# # Now create new grid of DM values to de-disperse
# dm_grid = np.arange(0., 1000., 50.)
# # Going through all pipeline because even de-dispersion parameters have
# # changed
# candidates = searcher.run(de_disp_func=de_disperse_cumsum,
#                           search_func=search_candidates,
#                           preprocess_func=create_ellipses,
#                           de_disp_args=[dm_grid],
#                           de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256},
#                           search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
#                                          'd_dm': 50.},
#                           preprocess_kwargs={'disk_size': 3,
#                                              'threshold_perc': 95.,
#                                              'statistic': 'mean'})
# print "Found {} pulses".format(len(candidates))
# for candidate in candidates:
#     print candidate
#
# # Going through all pipeline & using cached de-dispersed and pre-processed
# # values (because now using de-dispersion & pre-processing parameters that were
# # cached before and only search parameters have changed (searching not cached)
# candidates = searcher.run(de_disp_func=de_disperse_cumsum,
#                           search_func=search_candidates,
#                           preprocess_func=create_ellipses,
#                           de_disp_args=[dm_grid],
#                           de_disp_kwargs={'nu_max': 1684., 'd_nu': 16./256},
#                           preprocess_kwargs={'disk_size': 3,
#                                              'threshold_perc': 95.,
#                                              'statistic': 'mean'},
#                           search_kwargs={'n_d_x': 9., 'n_d_y': 17.,
#                                          'd_dm': 50.})
# print "Found {} pulses".format(len(candidates))
# for candidate in candidates:
#     print candidate
