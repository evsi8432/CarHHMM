import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

import sys

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.special import expit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

import divebomb

import time
import pickle

import Preprocessor
import Parameters
import HHMM
import Visualisor

rand_seed = int(sys.argv[1])
model = str(sys.argv[2])

np.random.seed(rand_seed)

# set parameters
pars = Parameters.Parameters()

pars.cvc_file = '../Data/2019/20190902-182840-CATs_OB_1.cvc'
pars.csv_file = '../Data/2019/20190902-182840-CATs_OB_1_001.csv'

pars.features = [{'dive_duration':{'corr':False,'f':'gamma'}},
                 {'Ahat_low':{'thresh':5,'corr':False,'f':'gamma'},
                  'Ax':{'corr':True,'f':'normal'},
                  'Ay':{'corr':True,'f':'normal'},
                  'Az':{'corr':True,'f':'normal'}}]
pars.K = [2,3]
pars.share_fine_states = True

if model == 'CarHMM':
    pars.K = [1,3]
elif model == 'HHMM':
    pars.features[1]['Ax']['corr'] = False
    pars.features[1]['Ay']['corr'] = False
    pars.features[1]['Az']['corr'] = False
elif model == 'CarHHMM1':
    pars.features[1] = {'Ax':{'corr':True,'f':'normal'},
                        'Ay':{'corr':True,'f':'normal'},
                        'Az':{'corr':True,'f':'normal'}}

# define files
HHMM_file = '../Params/%s_k_%s_%s_%s' % (model,pars.K[0],pars.K[1],rand_seed)
data_outfile = '../Params/data_%s_k_%s_%s_%s' % (model,pars.K[0],pars.K[1],rand_seed)

# preprocess data
prep = Preprocessor.Preprocessor(pars)
print('loading data')
df = prep.load_data(pars.cvc_file,pars.csv_file,pars.cvc_cols)
print('pruning cols')
df = prep.prune_cols(df)
print('pruning times')
df = prep.prune_times(df,pars.stime,pars.etime,pars.drop_times)
print('fixing pressure')
df = prep.fix_pressure(df)
print('finding Vz')
df = prep.find_Vz(df)
print('smoothing cols')
df = prep.smooth_columns(df,pars.smoother,pars.smooth_cols)
print('dividing into dives')
df,dive_df = prep.find_dives(df)
print('getting features')
data = prep.get_all_features(df,dive_df)

# train_model
print('training model')
hhmm = HHMM.HHMM(pars,data)
hhmm.train_DM(data,max_iters=10,max_steps=50)

# get SEs
h = 0.01
print('finding standard errors')
hhmm.get_SEs(data,h)

# label data
print('labelling df')
data,df = hhmm.label_df(data,df)

# save everything
hhmm.save(HHMM_file)
with open(data_outfile, 'wb') as f:
    pickle.dump(data, f)
