import numpy as np
from astropy.time import Time, TimeDelta
from frb.frames import create_from_txt
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import (search_candidates, search_candidates_ell,
                        search_candidates_clf, create_ellipses)
from frb.ml import PulseClassifier


print "Loading dynamical spectra"
txt = '/home/ilya/code/akutkin/frb/data/100_sec_wb_raes08a_128ch.asc'
frame = create_from_txt(txt, 1684., 0, 16./128, 0.001)
t0 = Time.now()
print "Start time {}".format(t0)
# Number of artificially injected pulses
n_pulses = 20
# Step of de-dispersion
d_dm = 30.
print "Adding {} pulses".format(n_pulses)

# Set random generator seed for reproducibility
np.random.seed(123)
# Generate values of pulse parameters
amps = np.random.uniform(0.15, 0.25, size=n_pulses)
widths = np.random.uniform(0.001, 0.003, size=n_pulses)
dm_values = np.random.uniform(100, 500, size=n_pulses)
times = np.linspace(0.1, frame.shape[0] - 0.1, n_pulses)
# Injecting pulses
for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
    frame.add_pulse(t_0, amp, width, dm)
    t_1 = t0 + TimeDelta(t_0, format='sec')
    print "Adding pulse with" \
          " t0={:%Y-%m-%d %H:%M:%S.%f},".format(t_1.utc.datetime)[:-3] +\
          " amp={:.2f}, width={:.4f}, dm={:.0f}".format(amp, width, dm)

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
                                         'amplitude': 3.,
                                         'save_fig': True},
                          preprocess_kwargs={'disk_size': 3,
                                             'threshold_big_perc': 97.5,
                                             'threshold_perc': 98.5,
                                             'statistic': 'mean'})
print "Found {} candidates".format(len(candidates))
for candidate in candidates:
    print candidate

print "using ``search_candidates_clf`` search function..."
# ICreate classifier class instance
pclf = PulseClassifier(de_disperse_cumsum, create_ellipses,
                       de_disp_args=[dm_grid],
                       preprocess_kwargs={'disk_size': 3,
                                          'threshold_big_perc': 97.5,
                                          'threshold_perc': 97.5,
                                          'statistic': 'mean'},
                       clf_kwargs={'kernel': 'rbf', 'probability': True,
                                   'class_weight': 'balanced'})
dsp = create_from_txt(txt, 1684., 0, 16./128, 0.001)
dsp = dsp.slice(0.2, 0.5)
t0 = Time.now()
dsp.meta_data = {'antenna': 'WB', 'freq': 'L', 'band': 'U', 'pol': 'R',
                 'exp_code': 'raks00', 'nu_max': 1684., 't_0': t0,
                 'd_nu': 16./128., 'd_t': 0.001}

# Generate values of pulses in training sample
print "Creating training sample"
n_training_pulses = 50
amps = np.random.uniform(0.15, 0.25, size=n_training_pulses)
widths = np.random.uniform(0.001, 0.003, size=n_training_pulses)
dm_values = np.random.uniform(100, 500, size=n_training_pulses)
times = np.linspace(0, 30, n_training_pulses+2)[1: -1]
features_dict, responses_dict = pclf.create_samples(dsp, amps, dm_values,
                                                    widths)
# print "Training classifier"
pclf.train(features_dict, responses_dict)

print "Searching FRBs in actual data"
# Note using the same arguments as in training classifier
candidates = searcher.run(de_disp_func=pclf.de_disp_func,
                          search_func=search_candidates_clf,
                          preprocess_func=pclf.preprocess_func,
                          de_disp_args=pclf.de_disp_args,
                          preprocess_kwargs=pclf.preprocess_kwargs,
                          search_args=[pclf],
                          search_kwargs={'d_dm': d_dm,
                                         'save_fig': True})
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
