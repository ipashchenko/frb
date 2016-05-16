import numpy as np
from astropy.time import Time
from frb.frames import Frame
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import search_candidates, create_ellipses
from frb.queries import query_frb, connect_to_db


# Set random generator seed for reproducibility
np.random.seed(123)

exp_code = 'raks00'
antennas = ['AR', 'EF', 'RA']
# Number of artificially injected pulses
n_pulses = 10
# Step of de-dispersion
d_dm = 25.

# Zero time
t = Time.now()
print "Zero time: {}".format(t)
# Time of `real` FRB
t_0_real = np.random.uniform(0, 10., size=1)[0]
amp_real = np.random.uniform(0.15, 0.20, size=1)[0]
width_real = np.random.uniform(0.001, 0.005, size=1)[0]
dm_value_real = np.random.uniform(0, 1000, size=1)[0]

print "REAL FRB: t={}, A={}, W={}, DM={}".format(t_0_real, amp_real, width_real,
                                                 dm_value_real)

for antenna in antennas:
    print "Creating Dynamical Spectra for antenna {}".format(antenna)
    frame = Frame(256, 10000, 1684., 0., 16./256, 1./1000)
    print "Adding {} pulses".format(n_pulses)
    # Generate values of pulse parameters
    amps = np.random.uniform(0.1, 0.15, size=n_pulses)
    widths = np.random.uniform(0.001, 0.005, size=n_pulses)
    dm_values = np.random.uniform(0, 1000, size=n_pulses)
    times = np.random.uniform(0., 10., size=n_pulses)
    # Injecting pulses
    for t_0, amp, width, dm in zip(times, amps, widths, dm_values):
        frame.add_pulse(t_0, amp, width, dm)
        print "Adding pulse with t0={}, amp={}, width={}," \
              " dm={}".format(t_0, amp, width, dm)
    print "Adding noise"
    frame.add_noise(0.5)
    print "Adding REAL FRB"
    frame.add_pulse(t_0_real, amp_real, width_real, dm_value_real)

    meta_data = {'antenna': antenna, 'freq': 'L', 'band': 'U', 'pol': 'R',
                 'exp_code': 'raks00', 'nu_max': 1684., 't_0': t,
                 'd_nu': 16./256., 'd_t': 0.001}
    # Values of DM to de-disperse
    dm_grid = np.arange(0., 1000., d_dm)
    # Initialize searcher class
    searcher = Searcher(dsp=frame.values, meta_data=meta_data)
    # Run search for FRB with some parameters of de-dispersion, pre-processing,
    # searching algorithms
    candidates = searcher.run(de_disp_func=de_disperse_cumsum,
                              search_func=search_candidates,
                              preprocess_func=create_ellipses,
                              de_disp_args=[dm_grid],
                              search_kwargs={'n_d_x': 5., 'n_d_y': 15.,
                                             'd_dm': d_dm},
                              preprocess_kwargs={'disk_size': 3,
                                                 'threshold_perc': 98.,
                                                 'statistic': 'mean'},
                              db_file='/home/ilya/code/akutkin/frb/frb/frb.db')
    print "Found {} pulses".format(len(candidates))
    for candidate in candidates:
        print candidate


session = connect_to_db("/home/ilya/code/akutkin/frb/frb/frb.db")
# Query DB
frb_list = query_frb(session, exp_code, d_dm=100., d_t=0.1)
for frb in frb_list:
    print frb

