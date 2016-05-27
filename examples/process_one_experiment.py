import os
from frb.cfx import CFX
from frb.utils import find_file
from frb.raw_data import M5, dspec_cat
from frb.search_candidates import Searcher
from frb.dedispersion import de_disperse_cumsum
from frb.search import search_candidates, create_ellipses
from frb.queries import query_frb, connect_to_db


# Setup
exp_code = 'raks12er'
cfx_file = '/home/ilya/code/frb/frb/RADIOASTRON_RAKS12ER_L_20151105T130000_ASC_V1.cfx'
data_dir = '/mnt/frb_data/raw_data/2015_309_raks12er'
dspec_params = {'nchan': 64, 'dt': 1, 'offst': 0, 'dur': 10, 'outfile': None}
# Split an M5-file into [sec] intervals
split_duration = 0.5

cobj = CFX(cfx_file)
cfx_data = cobj.parse_cfx(exp_code)
if cobj.freq == 'K':
    print("Skipping K-band CFX file: {}".format(os.path.basename(cfx_file)))
    print("NOTE: You can delete following files from data path:")
    print(cfx_data)
for fname, params in cfx_data.items():
    fname = fname.split(".")[0]
    import glob
    m5file = glob.glob(os.path.join(os.path.join(data_dir, params[1].lower()),
                                    fname + "*"))[0]
    m5file_fmt = params[2] # Raw data format
    cfx_fmt = params[-1]   # Rec configuration
    m5 = M5(m5file, m5file_fmt)
    offst = 0
    dspec_params.update({'dur': split_duration})
    while offst*32e6 < m5.size:
        dspec_params.update({'offst':offst})
        #                print dspec_params

        ds = m5.create_dspec(**dspec_params)
        # NOTE: all 4 channels are stacked forming dsarr:
        dsarr = dspec_cat(os.path.basename(ds['Dspec_file']), cfx_fmt)
        metadata = ds
        metadata['Raw_data_file'] = fname
        metadata['Exp_data'] = params
        print "BRV SEARCHING..."  # search brv in array here

        # TODO: save search results, delete data, ...
        offst = offst + split_duration
antennas = list()
antennas = ['AR', 'EF', 'RA']
# Step of de-dispersion
d_dm = 25.


for antenna in antennas:
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
