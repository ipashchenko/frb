# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from astropy.time import Time, TimeDelta
path = os.path.normpath(os.path.join(os.path.dirname(sys.argv[0]), '..'))
sys.path.insert(0, path)
from frb.dyn_spectra import create_from_txt
from frb.search_candidates import Searcher
from frb.dedispersion import noncoherent_dedisperse
from frb.search import (search_candidates, search_candidates_ell,
                        search_candidates_clf, create_ellipses)
from frb.ml import PulseClassifier

# TODO: Automatically find amplitudes of injected pulses used in training of
# classifiers
# TODO: Where should i train classifiers?

from frb.candidates import db_file
print "DB file {}".format(db_file)
print "Loading dynamical spectra"
txt = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                   '100_sec_wb_raes08a_128ch.asc')
meta_data = {'antenna': 'WB', 'freq': 'L', 'band': 'U', 'pol': 'R',
             'exp_code': 'raks00'}
t0 = Time.now()
dsp = create_from_txt(txt, 1684., 16. / 128, 0.001, meta_data, t0)
dsp = dsp.slice(0.8, 1)
print "Start time {}".format(t0)
# Number of artificially injected pulses
n_pulses = 3
# Step of de-dispersion
d_dm = 30.
print "Adding {} pulses".format(n_pulses)

# Set random generator seed for reproducibility
np.random.seed(123)
# Generate values of pulse parameters
amps = np.random.uniform(0.15, 0.25, size=n_pulses)
widths = np.random.uniform(0.001, 0.003, size=n_pulses)
dm_values = np.random.uniform(100, 500, size=n_pulses)
times = np.linspace(0.1, dsp.shape[0] - 0.1, n_pulses)
# Injecting pulses
for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
    dsp.add_pulse(t_0, amp, width, dm)
    t_1 = t0 + TimeDelta(t_0, format='sec')
    print "Adding pulse with" \
          " t0={:%Y-%m-%d %H:%M:%S.%f},".format(t_1.utc.datetime)[:-3] +\
          " amp={:.2f}, width={:.4f}, dm={:.0f}".format(amp, width, dm)

# Values of DM to de-disperse
dm_grid = np.arange(0., 1000., d_dm)

# Initialize searcher class
searcher = Searcher(dsp)

# # Run search for FRB with some parameters of de-dispersion, pre-processing,
# # searching algorithms
# print "using ``search_candidates`` search function..."
# candidates = searcher.run(de_disp_func=noncoherent_dedisperse,
#                           search_func=search_candidates,
#                           preprocess_func=create_ellipses,
#                           de_disp_args=[dm_grid],
#                           de_disp_kwargs={'threads': 4},
#                           search_kwargs={'n_d_x': 4., 'n_d_y': 15.,
#                                          'd_dm': d_dm},
#                           preprocess_kwargs={'disk_size': 3,
#                                              'threshold_big_perc': 97.5,
#                                              'threshold_perc': 98.5,
#                                              'statistic': 'mean'})
# print "Found {} candidates".format(len(candidates))
# for candidate in candidates:
#     print candidate
#
# Run search for FRB with same parameters of de-dispersion, but different
# pre-processing & searching algorithms
print "using ``search_candidates_ell`` search function..."
candidates = searcher.run(de_disp_func=noncoherent_dedisperse,
                          search_func=search_candidates_ell,
                          preprocess_func=create_ellipses,
                          de_disp_args=[dm_grid],
                          de_disp_kwargs={'threads': 4},
                          search_kwargs={'x_stddev': 6., 'y_to_x_stddev': 0.3,
                                         'theta_lims': [130., 180.],
                                         'x_cos_theta': 3.,
                                         'd_dm': d_dm,
                                         'amplitude': None,
                                         'save_fig': True},
                          preprocess_kwargs={'disk_size': 3,
                                             'threshold_big_perc': 97.5,
                                             'threshold_perc': 95.5,
                                             'statistic': 'mean'},
                          db_file=db_file)
print "Found {} candidates".format(len(candidates))
for candidate in candidates:
    print candidate

# print "using ``search_candidates_clf`` search function with SVM..."
# # ICreate classifier class instance
# pclf = PulseClassifier(de_disperse_cumsum, create_ellipses,
#                        de_disp_args=[dm_grid],
#                        preprocess_kwargs={'disk_size': 3,
#                                           'threshold_big_perc': 97.5,
#                                           'threshold_perc': 97.5,
#                                           'statistic': 'mean'},
#                        clf_kwargs={'kernel': 'rbf', 'probability': True,
#                                    'class_weight': 'balanced'})
# dsp_training = create_from_txt(txt, 1684., 16. / 128, 0.001, meta_data,
#                                t_0=Time.now())
# dsp_training = dsp_training.slice(0.2, 0.5)
#
# # Generate values of pulses in training sample
# print "Creating training sample"
# n_training_pulses = 50
# amps = np.random.uniform(0.15, 0.25, size=n_training_pulses)
# widths = np.random.uniform(0.001, 0.003, size=n_training_pulses)
# dm_values = np.random.uniform(100, 500, size=n_training_pulses)
# times = np.linspace(0, 30, n_training_pulses+2)[1: -1]
# features_dict, responses_dict = pclf.create_samples(dsp_training, amps,
#                                                     dm_values, widths)
# # print "Training classifier"
# pclf.train(features_dict, responses_dict)
#
# print "Searching FRBs in actual data"
# # Note using the same arguments as in training classifier
# candidates = searcher.run(de_disp_func=pclf.de_disp_func,
#                           search_func=search_candidates_clf,
#                           preprocess_func=pclf.preprocess_func,
#                           de_disp_args=pclf.de_disp_args,
#                           preprocess_kwargs=pclf.preprocess_kwargs,
#                           search_args=[pclf],
#                           search_kwargs={'d_dm': d_dm,
#                                          'save_fig': True})
# print "Found {} candidates".format(len(candidates))
# for candidate in candidates:
#     print candidate

print "using ``search_candidates_clf`` search function with GBC..."
# ICreate classifier class instance
from sklearn.ensemble import GradientBoostingClassifier
pclf = PulseClassifier(noncoherent_dedisperse, create_ellipses,
                       clf=GradientBoostingClassifier,
                       clf_kwargs={'verbose': 0, 'n_estimators': 3000},
                       de_disp_args=[dm_grid],
                       de_disp_kwargs={'threads': 4},
                       preprocess_kwargs={'disk_size': 3,
                                          'threshold_big_perc': 97.5,
                                          'threshold_perc': 97.5,
                                          'statistic': 'mean'})
dsp_training = create_from_txt(txt, 1684., 16. / 128, 0.001, meta_data,
                               t_0=Time.now())
dsp_training = dsp_training.slice(0.2, 0.8)

# Generate values of pulses in training sample
print "Creating training sample"
n_training_pulses = 100
amps = np.random.uniform(0.15, 0.25, size=n_training_pulses)
widths = np.random.uniform(0.001, 0.003, size=n_training_pulses)
dm_values = np.random.uniform(100, 500, size=n_training_pulses)
times = np.linspace(0, 30, n_training_pulses+2)[1: -1]
features_dict, responses_dict = pclf.create_samples(dsp_training, amps,
                                                    dm_values, widths)
# print "Training classifier"
pclf.train(features_dict, responses_dict)

print "Searching FRBs in actual data"
# Note using the same arguments as in training classifier
candidates = searcher.run(de_disp_func=pclf.de_disp_func,
                          search_func=search_candidates_clf,
                          preprocess_func=pclf.preprocess_func,
                          de_disp_args=pclf.de_disp_args,
                          de_disp_kwargs=pclf.de_disp_kwargs,
                          preprocess_kwargs=pclf.preprocess_kwargs,
                          search_args=[pclf],
                          search_kwargs={'d_dm': d_dm,
                                         'save_fig': True},
                          db_file=db_file)
print "Found {} candidates".format(len(candidates))
for candidate in candidates:
    print candidate
