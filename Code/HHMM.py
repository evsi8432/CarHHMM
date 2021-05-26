import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import vonmises
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import circstd
from scipy.special import iv
from scipy.special import expit
from scipy.special import logit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

from skopt import gp_minimize

from copy import deepcopy

import divebomb

import time
import pickle


def corr_2_rho(corr):
    return logit(corr)

def ptm_2_eta(gamma):

    eta = np.zeros_like(gamma)
    N = len(gamma)

    for _ in range(100):
        for i in range(N):
            A = sum(np.exp(eta[i]))
            for j in range(N):
                if i != j:
                    A0 = A-np.exp(eta[i,j])
                    eta[i,j] = np.log((A0*gamma[i,j])/(1-gamma[i,j]))

    return eta

def eta_2_ptm(eta):
    ptm = np.exp(eta)
    return (ptm.T/np.sum(ptm,1)).T


class HHMM:


    def __init__(self,pars,data):

        self.pars = pars
        self.theta = []
        self.eta = []
        self.ptm = []


        self.data = data
        self.SEs = None
        self.train_time = None
        self.CM = None

        eta_crude = -2.0 + np.random.normal(size=(self.pars.K[0],self.pars.K[0]))
        ptm_crude = np.exp(eta_crude)
        ptm_crude = (ptm_crude.T/np.sum(ptm_crude,1)).T

        self.eta.append(eta_crude)
        self.ptm.append(ptm_crude)

        eta_fine = []
        ptm_fine = []
        for _ in range(self.pars.K[0]):
            eta_fine_k = -2.0 + np.random.normal(size=(self.pars.K[1],self.pars.K[1]))
            ptm_fine_k = np.exp(eta_fine_k)
            ptm_fine_k = (ptm_fine_k.T/np.sum(ptm_fine_k,1)).T
            eta_fine.append(eta_fine_k)
            ptm_fine.append(ptm_fine_k)

        self.eta.append(eta_fine)
        self.ptm.append(ptm_fine)

        self.initialize_theta(data)

        self.true_theta = None
        self.true_eta = None

        return


    def logdotexp(self, A, B):
        max_A = np.max(A)
        max_B = np.max(B)
        C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
        np.log(C, out=C)
        C += max_A + max_B
        return C


    def initialize_theta(self,data):

        theta = []

        # first fill in the dive level values
        K = self.pars.K[0]
        theta.append({})
        for feature,settings in self.pars.features[0].items():

            # initialize values
            theta[0][feature] = {'mu': np.zeros(K),
                                 'sig': np.zeros(K),
                                 'corr': np.zeros(K)}


            if data is not None:

                feature_data = [datum[feature] for datum in data]

                # first find mu
                theta[0][feature]['mu'] = np.mean(feature_data)*np.ones(K)

                # then get varaince of each quantile set of data
                data_sorted = np.sort(feature_data)
                n = len(data_sorted)
                for k0 in range(K):
                    if settings['f'] != 'vonmises':
                        std = np.std(data_sorted[int(k0*n/K):int((k0+1)*n/K)])
                        theta[0][feature]['sig'][k0] = max(0.1,std)
                    else:
                        theta[0][feature]['sig'][k0] = 1.0

                # initialize correlations
                theta[0][feature]['corr'] = -1.0 * np.ones(K)

                # add randomness in initialization
                if settings['f'] == 'normal':
                    theta[0][feature]['mu'] += norm.rvs(np.zeros(K),theta[0][feature]['sig'])
                    theta[0][feature]['sig'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                    theta[0][feature]['corr'] += norm.rvs(0.0,1.0,size=K)
                elif settings['f'] == 'gamma':
                    theta[0][feature]['mu'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                    theta[0][feature]['sig'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                    theta[0][feature]['corr'] += norm.rvs(0.0,1.0,size=K)
                else:
                    pass

        # then fill in the subdive level values
        theta.append([{} for _ in range(K)])
        K = self.pars.K[1]

        for feature,settings in self.pars.features[1].items():
            for k0 in range(self.pars.K[0]):

                # initialize values
                theta[1][k0][feature] = {'mu': np.zeros(K),
                                         'sig': np.zeros(K),
                                         'corr': np.zeros(K)}

                if data is not None:

                    feature_data = []
                    for dive in data:
                        feature_data.extend([seg[feature] for seg in dive['subdive_features']])

                    # first find mu
                    theta[1][k0][feature]['mu'] = np.mean(feature_data)*np.ones(K)

                    # then get varaince of each quantile set of data
                    data_sorted = np.sort(feature_data)
                    n = len(data_sorted)
                    theta[1][k0][feature]['sig'] = np.std(feature_data)*np.ones(K)

                    # finally update correlations randomly
                    theta[1][k0][feature]['corr'] = -1.0 * np.ones(K)

                    # add randomness in initialization
                    if settings['f'] == 'normal':
                        theta[1][k0][feature]['mu'] += norm.rvs(np.zeros(K),theta[1][k0][feature]['sig'])
                        theta[1][k0][feature]['sig'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                        theta[1][k0][feature]['corr'] += norm.rvs(0.0,1.0,size=K)
                    elif settings['f'] == 'gamma':
                        theta[1][k0][feature]['mu'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                        theta[1][k0][feature]['sig'] *= np.exp(norm.rvs(0.0,1.0,size=K))
                        theta[1][k0][feature]['corr'] += norm.rvs(0.0,1.0,size=K)
                    else:
                        pass

        self.theta = theta

        return


    def find_log_p_yt_given_xt(self,level,feature,data,data_tm1,mu,sig,corr,sample=0):

        # find log density of feature
        if self.pars.features[level][feature]['f'] == 'multivariate_normal':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
            elif self.pars.features[level][feature]['corr'] and data_tm1 is None:
                return 0

            if sample > 0:
                return multivariate_normal.rvs(mu,sig,sample)
            else:
                return multivariate_normal.logpdf(mu,data[feature],sig)

        # find log density of feature
        if self.pars.features[level][feature]['f'] == 'normal':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
            elif self.pars.features[level][feature]['corr'] and data_tm1 is None:
                return 0

            if sample > 0:
                return norm.rvs(mu,sig,sample)
            else:
                return norm.logpdf(data[feature],mu,sig)

        elif self.pars.features[level][feature]['f'] == 'gamma':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
            elif self.pars.features[level][feature]['corr'] and data_tm1 is None:
                return 0

            shape = np.square(mu)/np.square(sig)
            scale = np.square(sig)/np.array(mu)
            if sample > 0:
                return gamma.rvs(shape,0,scale,sample)
            else:
                return gamma.logpdf(data[feature],shape,0,scale)

        else:

            # find new mean if there is autocorrelation
            if (self.pars.features[1][feature]['corr']) and data_tm1 is not None and (mu < 0) and (data_tm1[feature] > mu+np.pi) and (data_tm1[feature] < np.pi):
                mu += 2*np.pi
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
                mu = ((mu+np.pi)%(2*np.pi))-np.pi
            elif (self.pars.features[1][feature]['corr']) and data_tm1 is not None and (data_tm1[feature] < 0) and (mu > data_tm1[feature]+np.pi) and (mu < np.pi):
                data_tm1[feature] += 2*np.pi
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
                mu = ((mu+np.pi)%(2*np.pi))-np.pi
            elif self.pars.features[1][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]
            elif self.pars.features[level][feature]['corr'] and data_tm1 is None:
                return 0
            else:
                pass

            kappa = sig

            if sample > 0:
                return vonmises.rvs(kappa,loc=mu,size=sample)
            else:
                return vonmises.logpdf(data[feature],kappa,loc=mu)


    def dive_likelihood(self,dive_data,state):

        # deal with dive lengths of 0
        if len(dive_data) == 0:
            return 0

        # find tpm
        self.eta[1][state][np.diag_indices(self.pars.K[1])] = 0
        ptm = np.exp(self.eta[1][state])
        ptm = (ptm.T/np.sum(ptm,1)).T
        log_ptm = np.log(ptm)

        # find the initial distribution (stationary distribution)
        delta = np.ones((1,self.pars.K[1]))/self.pars.K[1]
        for _ in range(10):
            delta = delta.dot(ptm)
        log_phi = np.log(delta)

        # initialize values
        log_L = 0
        seg_tm1 = None

        # iterate through dive segments
        for seg in dive_data:

            log_p_yt_given_xt = np.zeros(self.pars.K[1])

            # find likelihood for each feature
            for feature in self.pars.features[1]:
                mu = np.copy(self.theta[1][state][feature]['mu'])
                sig = np.copy(self.theta[1][state][feature]['sig'])
                corr = np.copy(expit(self.theta[1][state][feature]['corr']))
                log_p_yt_given_xt += self.find_log_p_yt_given_xt(1,feature,
                                                                 seg,seg_tm1,
                                                                 mu,sig,corr)

            # update transition
            log_v = self.logdotexp(log_phi,log_ptm) + log_p_yt_given_xt
            log_u = logsumexp(log_v)
            log_L += log_u
            log_phi = log_v - log_u
            seg_tm1 = seg

        return log_L


    def likelihood(self,data):

        # find tpm
        self.eta[0][np.diag_indices(self.pars.K[0])] = 0
        ptm = np.exp(self.eta[0])
        ptm = (ptm.T/np.sum(ptm,1)).T
        log_ptm = np.log(ptm)

        # find the initial distribution (stationary distribution)
        delta = np.ones((1,self.pars.K[0]))/self.pars.K[0]
        for _ in range(10):
            delta = delta.dot(ptm)
        log_phi = np.log(delta)

        # initialize values
        log_L = 0
        dive_tm1 = None

        # iterate through dive segments
        for dive in data:

            # initialize values
            log_p_yt_given_xt = np.zeros(self.pars.K[0])

            # find likelihood for each feature
            for feature in self.pars.features[0]:
                mu = np.copy(self.theta[0][feature]['mu'])
                sig = np.copy(self.theta[0][feature]['sig'])
                corr = np.copy(expit(self.theta[0][feature]['corr']))
                log_p_yt_given_xt += self.find_log_p_yt_given_xt(0,feature,
                                                                 dive,dive_tm1,
                                                                 mu,sig,corr)

            # find likelihood of subdive features:
            for k0 in range(self.pars.K[0]):
                log_p_yt_given_xt[k0] += self.dive_likelihood(dive['subdive_features'],k0)

            # update transition
            log_v = self.logdotexp(log_phi,log_ptm) + log_p_yt_given_xt
            log_u = logsumexp(log_v)
            log_L += log_u
            log_phi = log_v - log_u
            dive_tm1 = dive

        return log_L


    def fwd_bwd(self,data,level):

        K = self.pars.K[level[0]]
        T = len(data)

        # find the tpm
        if level[0] == 0:
            self.eta[0][np.diag_indices(K)] = 0
            ptm = np.exp(self.eta[0])
            ptm = (ptm.T/np.sum(ptm,1)).T
            #ptm = self.ptm[0]
            log_ptm = np.log(ptm)
        else:
            self.eta[1][level[1]][np.diag_indices(K)] = 0
            ptm = np.exp(self.eta[1][level[1]])
            ptm = (ptm.T/np.sum(ptm,1)).T
            #ptm = self.ptm[1][level[1]]
            log_ptm = np.log(ptm)

        # find the initial distribution (stationary distribution)
        delta = np.ones((1,K))/K
        for _ in range(10):
            delta = delta.dot(ptm)
        log_delta = np.log(delta)

        # overall likelihood
        L_alpha = 0
        L_beta = 0

        # initialize values
        log_alpha = np.zeros((K,T))
        log_beta = np.zeros((K,T))
        log_beta[:,-1] = np.ones(K)

        dive_tm1 = None

        # first, store log_p_yt_given_xt:
        log_p_yt_given_xt = np.zeros((K,T))

        for t,dive in enumerate(data):

            # find likelihood for each feature
            if level[0] == 0:
                for feature in self.pars.features[0]:
                    mu = self.theta[0][feature]['mu']
                    sig = self.theta[0][feature]['sig']
                    corr = expit(self.theta[0][feature]['corr'])
                    log_p_yt_given_xt[:,t] += self.find_log_p_yt_given_xt(0,feature,
                                                                          dive,dive_tm1,
                                                                          mu,sig,corr)
            else:
                state = level[1]
                for feature in self.pars.features[1]:
                    mu = self.theta[1][state][feature]['mu']
                    sig = self.theta[1][state][feature]['sig']
                    corr = expit(self.theta[1][state][feature]['corr'])
                    log_p_yt_given_xt[:,t] += self.find_log_p_yt_given_xt(1,feature,
                                                                          dive,dive_tm1,
                                                                          mu,sig,corr)

            # find likelihood of subdive features:
            if level[0] == 0:
                for k0 in range(K):
                    log_p_yt_given_xt[k0,t] += self.dive_likelihood(dive['subdive_features'],k0)

            # update previous dive
            dive_tm1 = dive

        # forward algorithm
        for t,dive in enumerate(data):

            # add log-likelihood and adjust for vanishing gradients
            if t == 0:
                log_alpha[:,t] = log_delta + log_p_yt_given_xt[:,t]
            else:
                log_alpha[:,t] = self.logdotexp(log_alpha[:,t-1],log_ptm) + log_p_yt_given_xt[:,t]

        L_alpha = logsumexp(log_alpha[:,-1])

        # backward algorithm
        for t,y_t in enumerate(reversed(data)):

            # add log-likelihood and adjust for vanishing gradients
            if t == 0:
                log_beta[:,-t-1] = 0
            else:
                log_beta[:,-t-1] = self.logdotexp(log_ptm + log_p_yt_given_xt[:,-t],
                                                  log_beta[:,-t])

        L_beta = logsumexp(log_beta[:,0])

        # find posterior (gamma)
        log_gamma = np.zeros((K,T))
        for t in range(T):
            log_gamma[:,t] = log_alpha[:,t] + log_beta[:,t]
            log_gamma[:,t] = log_gamma[:,t] - logsumexp(log_gamma[:,t])
        gamma = np.exp(log_gamma)

        # find xi
        xi = np.zeros((K,K,T-1))
        log_xi = np.zeros((K,K,T-1))
        for t in range(T-1):
            log_xi[:,:,t] = (log_alpha[:,t] + log_ptm.T).T + log_beta[:,t+1]
            log_xi[:,:,t] = log_xi[:,:,t] + log_p_yt_given_xt[:,t+1]
            log_xi[:,:,t] = log_xi[:,:,t]-logsumexp(log_xi[:,:,t])
        xi = np.exp(log_xi)

        return log_alpha, log_beta, gamma, xi


    def train_GP(self,data,n_initial=10,n_calls=10,eps=10e-6):

        stime = time.time()
        prev_l = self.likelihood(data)

        def loss_fn(x,get_lims=False):

            ind = 0

            if get_lims:
                lims = []

            # update crude eta
            for i in range(self.pars.K[0]):
                for j in range(self.pars.K[0]-1):
                    if j < i:
                        self.eta[0][i,j] = x[ind]
                    else:
                        self.eta[0][i,j+1] = x[ind]

                    if get_lims:
                        lims.append((-10.0,10.0))

                    ind += 1

            # update fine eta
            for k0 in range(self.pars.K[0]):
                for i in range(self.pars.K[1]):
                    for j in range(self.pars.K[1]-1):
                        if j < i:
                            self.eta[1][k0][i,j] = x[ind]
                        else:
                            self.eta[1][k0][i,j+1] = x[ind]

                        if get_lims:
                            lims.append((-10.0,10.0))

                        ind += 1

            # update crude theta
            for k0 in range(self.pars.K[0]):
                for feature in self.pars.features[0]:
                    for param in ['mu','sig','corr']:
                        if feature == 'Ax' and param in ['corr']:
                            self.theta[0]['Ax'][param][k0] = x[ind]
                            self.theta[0]['Ay'][param][k0] = x[ind]
                            self.theta[0]['Az'][param][k0] = x[ind]
                        else:
                            if param == 'sig':
                                self.theta[0][feature][param][k0] = max(x[ind],eps)
                            else:
                                self.theta[0][feature][param][k0] = x[ind]

                        if get_lims:
                            feature_data = [datum[feature] for datum in data]
                            if param == 'mu':
                                lims.append((min(feature_data),max(feature_data)))
                            elif param == 'sig':
                                lims.append((eps,max(feature_data)-min(feature_data)))
                            else:
                                lims.append((-10.0,10.0))

                        ind += 1

            # update fine theta
            for k1 in range(self.pars.K[1]):

                if self.pars.share_fine_states:
                    K0 = 1
                else:
                    K0 = self.pars.K[0]

                for feature in self.pars.features[1]:
                    for k0 in range(K0):
                        for param in ['mu','sig','corr']:
                            for k00 in range(self.pars.K[0]):

                                # continue if we not sharing the states
                                if not self.pars.share_fine_states and (k00 != k0):
                                    continue

                                # set theta
                                if feature == 'Ax' and param in ['corr']:
                                    self.theta[1][k00]['Ax'][param][k1] = x[ind]
                                    self.theta[1][k00]['Ay'][param][k1] = x[ind]
                                    self.theta[1][k00]['Az'][param][k1] = x[ind]
                                else:
                                    if param == 'sig':
                                        self.theta[1][k00][feature][param][k1] = max(x[ind],eps)
                                    else:
                                        self.theta[1][k00][feature][param][k1] = x[ind]

                            if get_lims:
                                feature_data = []
                                for datum in data:
                                    for seg in datum['subdive_features']:
                                        feature_data.append(seg[feature])
                                if param == 'mu':
                                    lims.append((min(feature_data),max(feature_data)))
                                elif param == 'sig':
                                    lims.append((eps,max(feature_data)-min(feature_data)))
                                else:
                                    lims.append((-10.0,10.0))

                            ind += 1

            if get_lims:
                return lims
            else:
                return -self.likelihood(data)

        # get limits
        lims = loss_fn(np.ones(1000),get_lims=True)

        # optimize
        res = gp_minimize(loss_fn,lims,n_calls=n_calls)

        # set values
        loss_fn(res['x'])

        print(res['x'])
        print('')
        print(self.eta)
        print('')
        print(self.theta)

        return (self.theta,self.eta)

    def train_DM(self,data,max_iters=10,max_steps=10,tol=0.01,eps=10e-6):

        stime = time.time()
        options = {'maxiter':max_iters,'disp':False}
        prev_l = self.likelihood(data)

        for n in range(max_steps):

            print('ITERATION %d'%n)
            print(prev_l)
            print('')

            ### start with crude eta ###
            for i in range(self.pars.K[0]):
                backup = deepcopy(self.eta[0][i,:])
                backup_l = deepcopy(self.likelihood(data))

                def loss_fn(x):

                    # update crude eta
                    for j,xj in enumerate(x):
                        if j < i:
                            self.eta[0][i,j] = xj
                        else:
                            self.eta[0][i,j+1] = xj

                    return -self.likelihood(data)

                # define inital value
                x0 = deepcopy(self.eta[0][i,:])
                x0 = np.delete(x0,i)

                # optimize
                if len(x0) > 0:
                    res = minimize(loss_fn, x0, method='BFGS',options=options)

                    # update final values
                    x = np.copy(res['x'])
                    if self.likelihood(data) >= backup_l:
                        for j,xj in enumerate(x):
                            if j < i:
                                self.eta[0][i,j] = xj
                            else:
                                self.eta[0][i,j+1] = xj
                    else:
                        self.eta[0][i,:] = backup
                        print('DANGER')

                    print('updated coarse eta, row %d' % i)
                    print('original: ', backup)
                    print('new: ', x)
                    print(self.likelihood(data))
                else:
                    print('N = 1, no Coarse Gamma')


            #### then do crude theta ###
            for k0 in range(self.pars.K[0]):
                for feature in self.pars.features[0]:
                    for param in ['mu','sig','corr']:

                        backup = deepcopy(self.theta[0][feature][param][k0])
                        backup_l = deepcopy(self.likelihood(data))

                        # find when not to do the optimization
                        if param == 'corr' and not self.pars.features[0][feature]['corr']:
                            continue
                        if feature in ['Ay','Az'] and param in ['corr']:
                            continue

                        def loss_fn(x):
                            if feature == 'Ax' and param in ['corr']:
                                self.theta[0]['Ax'][param][k0] = x
                                self.theta[0]['Ay'][param][k0] = x
                                self.theta[0]['Az'][param][k0] = x
                            else:
                                if param == 'sig':
                                    self.theta[0][feature][param][k0] = max(x,eps)
                                else:
                                    self.theta[0][feature][param][k0] = x

                            return -self.likelihood(data)

                        # define inital value
                        x0 = self.theta[0][feature][param][k0]

                        # optimize
                        res = minimize(loss_fn, x0, method='Nelder-Mead', options=options)

                        # update final values
                        x = np.copy(res['x'])
                        if self.likelihood(data) >= backup_l:
                            if feature == 'Ax' and param in ['corr']:
                                self.theta[0]['Ax'][param][k0] = x
                                self.theta[0]['Ay'][param][k0] = x
                                self.theta[0]['Az'][param][k0] = x
                            else:
                                if param == 'sig':
                                    self.theta[0][feature][param][k0] = max(x,eps)
                                else:
                                    self.theta[0][feature][param][k0] = x
                        else:
                            self.theta[0][feature][param][k0] = backup
                            print('DANGER')

                        print('updated coarse theta %s, state %d, param %s' % (feature,k0,param))
                        print('original: ', backup)
                        print('new: ', x)
                        print(self.likelihood(data))


            ### then do fine eta ###
            for k0 in range(self.pars.K[0]):
                for i in range(self.pars.K[1]):

                    backup = deepcopy(self.eta[1][k0][i,:])
                    backup_l = self.likelihood(data)

                    def loss_fn(x):

                        # update crude eta
                        for j,xj in enumerate(x):
                            if j < i:
                                self.eta[1][k0][i,j] = xj
                            else:
                                self.eta[1][k0][i,j+1] = xj

                        return -self.likelihood(data)

                    # define inital value
                    x0 = deepcopy(self.eta[1][k0][i,:])
                    x0 = np.delete(x0,i)

                    # optimize
                    res = minimize(loss_fn, x0, method='BFGS',options=options)

                    # update final values
                    x = np.copy(res['x'])
                    if self.likelihood(data) >= backup_l:

                        for j,xj in enumerate(x):
                            if j < i:
                                self.eta[1][k0][i,j] = xj
                            else:
                                self.eta[1][k0][i,j+1] = xj
                    else:
                        self.eta[1][k0][i,:] = backup
                        print('DANGER')

                    print('updated fine eta, dive %d, row %d' % (k0,i))
                    print('original: ', backup)
                    print('new: ', x)
                    print(self.likelihood(data))


            ### finally do fine theta ###
            for k1 in range(self.pars.K[1]):

                if self.pars.share_fine_states:
                    K0 = 1
                else:
                    K0 = self.pars.K[0]

                for feature in self.pars.features[1]:
                    for k0 in range(K0):
                        for param in ['mu','sig','corr']:

                            # find when not to do the optimization
                            if param == 'corr' and not self.pars.features[1][feature]['corr']:
                                continue
                            if feature in ['Ay','Az'] and param in ['corr']:
                                continue

                            backup = deepcopy(self.theta[1][k0][feature][param][k1])
                            backup_l = deepcopy(self.likelihood(data))

                            def loss_fn(x):

                                for k00 in range(self.pars.K[0]):

                                    # continue if we not sharing the states
                                    if not self.pars.share_fine_states and (k00 != k0):
                                        continue

                                    # set theta
                                    if feature == 'Ax' and param in ['corr']:
                                        self.theta[1][k00]['Ax'][param][k1] = x
                                        self.theta[1][k00]['Ay'][param][k1] = x
                                        self.theta[1][k00]['Az'][param][k1] = x
                                    else:
                                        if param == 'sig':
                                            self.theta[1][k00][feature][param][k1] = max(x,eps)
                                        else:
                                            self.theta[1][k00][feature][param][k1] = x

                                return -self.likelihood(data)

                            # define inital value
                            x0 = self.theta[1][k0][feature][param][k1]

                            # optimize
                            res = minimize(loss_fn, x0, method='Nelder-Mead', options=options)

                            # update final values
                            if self.likelihood(data) >= backup_l:
                                x = np.copy(res['x'])
                            else:
                                x = backup
                                print('DANGER')

                            for k00 in range(self.pars.K[0]):

                                # continue if we not sharing the states
                                if not self.pars.share_fine_states and (k00 != k0):
                                    continue
                                # set theta
                                if feature == 'Ax' and param in ['corr']:
                                    self.theta[1][k00]['Ax'][param][k1] = x
                                    self.theta[1][k00]['Ay'][param][k1] = x
                                    self.theta[1][k00]['Az'][param][k1] = x
                                else:
                                    if param == 'sig':
                                        self.theta[1][k00][feature][param][k1] = max(x,eps)
                                    else:
                                        self.theta[1][k00][feature][param][k1] = x

                            print('updated fine theta %s, dive_state %d subdive state %d, param %s'\
                                  % (feature,k0,k1,param))
                            print('original: ', backup)
                            print('new: ', x)
                            print(self.likelihood(data))

            curr_l = self.likelihood(data)
            if abs(curr_l - prev_l) < tol:
                break
            else:
                prev_l = curr_l
            print('\n\n')

        self.train_time = time.time()-stime

        return (self.theta,self.eta)


    def label_df(self,data,df):

        # initalized dataframe state probs
        df['ML_subdive'] = -1
        df['ML_dive'] = -1
        for k0 in range(self.pars.K[0]):
            df['dive_state_' + str(k0) + '_prob'] = -1
        for k1 in range(self.pars.K[1]):
            df['subdive_state_' + str(k1) + '_prob'] = -1

        # get dive level posterior
        _,_,dive_post,_ = self.fwd_bwd(data,[0])

        # get subdive level posterior
        subdive_posts = []
        for dive_num,dive in enumerate(data):
            _,_,subdive_post,_ = self.fwd_bwd(dive['subdive_features'],[1,0])
            subdive_post *= dive_post[0,dive_num]
            for k0 in range(1,self.pars.K[0]):
                _,_,subdive_post_k0,_ = self.fwd_bwd(dive['subdive_features'],[1,k0])
                subdive_post += dive_post[k0,dive_num] * subdive_post_k0

            subdive_posts.append(subdive_post)

        # label the dive probs
        for dive_num,dive in enumerate(data):
            dive['dive_state_probs'] = dive_post[:,dive_num]
            for k0 in range(self.pars.K[0]):
                col = 'dive_state_' + str(k0) + '_prob'
                df[col][(df['time'] > dive['start_dive']) & \
                        (df['time'] < dive['end_dive'])] = dive_post[k0,dive_num]

            # put in most likely dive
            ML_dive = np.argmax(dive_post[:,dive_num])
            df['ML_dive'][(df['time'] > dive['start_dive']) & \
                          (df['time'] < dive['end_dive'])] = ML_dive

            # label the subdive_probs
            subdive_post = subdive_posts[dive_num]
            for seg_num,seg in enumerate(dive['subdive_features']):
                seg['subdive_state_probs'] = subdive_post[:,seg_num]
                for k1 in range(self.pars.K[1]):
                    col = 'subdive_state_' + str(k1) + '_prob'
                    df[col][(df['time'] > seg['start_time']) & \
                            (df['time'] < seg['end_time'])] = subdive_post[k1,seg_num]

                # put in most likely subdive
                ML_subdive = np.argmax(subdive_post[:,seg_num])
                df['ML_subdive'][(df['time'] > seg['start_time']) & \
                                 (df['time'] < seg['end_time'])] = ML_subdive

        return data,df


    def get_SEs(self,data,h):

        SEs = {}

        # coarse-scale theta
        for feature in self.theta[0]:
            print('')
            print(feature)
            SEs[feature] = {}
            for param in self.theta[0][feature]:
                print('')
                print(param)
                SEs[feature][param] = []
                for state_num,theta in enumerate(self.theta[0][feature][param]):
                    print(state_num)

                    if param == 'corr':
                        theta = deepcopy(expit(self.theta[0][feature][param][state_num]))
                        h0 = h*min(theta,1.0-theta)

                    # get middle value
                    th_t = self.likelihood(data)

                    # get plus value
                    if param == 'corr':
                        self.theta[0][feature][param][state_num] = corr_2_rho(theta+h0)
                    else:
                        self.theta[0][feature][param][state_num] += h
                    th_tp1 = self.likelihood(data)

                    # get minus value
                    if param == 'corr':
                        self.theta[0][feature][param][state_num] = corr_2_rho(theta-h0)
                    else:
                        self.theta[0][feature][param][state_num] += -2*h
                    th_tm1 = self.likelihood(data)

                    # return theta
                    if param == 'corr':
                        self.theta[0][feature][param][state_num] = corr_2_rho(theta)
                    else:
                        self.theta[0][feature][param][state_num] += h

                    # get estimate
                    if param =='corr':
                        I_th = (2*th_t - th_tm1 - th_tp1)/(h0**2)
                    else:
                        I_th = (2*th_t - th_tm1 - th_tp1)/(h**2)
                    V_th = 1.0/I_th
                    sig_th = np.sqrt(V_th)

                    SEs[feature][param].append(sig_th)


        # fine-scale theta (shared states)
        for feature in self.theta[1][0]:
            print('')
            print(feature)
            SEs[feature] = {}
            for param in self.theta[1][0][feature]:
                print('')
                print(param)
                SEs[feature][param] = []
                for state_num,theta in enumerate(self.theta[1][0][feature][param]):
                    print(state_num)

                    if param == 'corr':
                        theta = deepcopy(expit(self.theta[1][0][feature][param][state_num]))
                        h0 = h*min(theta,1.0-theta)

                    # get middle value
                    th_t = np.copy(self.likelihood(data))

                    # get plus value
                    if param == 'corr':
                        self.theta[1][0][feature][param][state_num] = corr_2_rho(theta+h0)
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]
                    else:
                        self.theta[1][0][feature][param][state_num] += h
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]
                    th_tp1 = np.copy(self.likelihood(data))

                    # get minus value
                    if param == 'corr':
                        self.theta[1][0][feature][param][state_num] = corr_2_rho(theta-h0)
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]
                    else:
                        self.theta[1][0][feature][param][state_num] += -2*h
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]
                    th_tm1 = np.copy(self.likelihood(data))

                    # print results
                    print(th_tm1)
                    print(th_t)
                    print(th_tp1)

                    # return theta
                    if param == 'corr':
                        self.theta[1][0][feature][param][state_num] = corr_2_rho(theta)
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]
                    else:
                        self.theta[1][0][feature][param][state_num] += h
                        for i in range(1,self.pars.K[0]):
                            self.theta[1][i][feature][param][state_num] = self.theta[1][0][feature][param][state_num]

                    # get estimate
                    if param =='corr':
                        I_th = (2*th_t - th_tm1 - th_tp1)/(h0**2)
                    else:
                        I_th = (2*th_t - th_tm1 - th_tp1)/(h**2)
                    V_th = 1.0/I_th
                    sig_th = np.sqrt(V_th)

                    SEs[feature][param].append(sig_th)


        # coarse-scale eta
        ptm = deepcopy(eta_2_ptm(self.eta[0]))

        V_gamma_coarse = np.zeros_like(self.eta[0])
        for i in range(self.eta[0].shape[0]):
            for j in range(self.eta[0].shape[1]):
                if i == j:
                    continue

                h0 = h*min(ptm[i,j],ptm[i,i])
                print(i,j)
                print(h0)

                # get middle value
                th_t = self.likelihood(data)

                # get plus value
                ptm[i,j] += h0
                ptm[i,i] -= h0
                self.eta[0] = ptm_2_eta(ptm)
                th_tp1 = self.likelihood(data)
                print(th_tp1)

                # get minus value
                ptm[i,j] -= 2*h0
                ptm[i,i] += 2*h0
                self.eta[0] = ptm_2_eta(ptm)
                th_tm1 = self.likelihood(data)
                print(th_tm1)

                # return theta
                ptm[i,j] += h0
                ptm[i,i] -= h0
                self.eta[0] = ptm_2_eta(ptm)

                # get estimate
                I_th = (2*th_t - th_tm1 - th_tp1)/(h0**2)
                V_th = 1.0/I_th
                V_gamma_coarse[i,j] = V_th
                print('')


        # get ptm
        SEs['Gamma_coarse'] = np.sqrt(V_gamma_coarse)

        # fine-scale eta
        V_gamma_fine = [np.zeros_like(x) for x in self.eta[1]]
        for n in range(len(V_gamma_fine)):
            ptm = deepcopy(eta_2_ptm(self.eta[1][n]))
            for i in range(self.eta[1][n].shape[0]):
                for j in range(self.eta[1][n].shape[1]):
                    if i == j:
                        continue

                    h0 = h*min(ptm[i,j],ptm[i,i])
                    print(i,j)
                    print(h0)

                    # get middle value
                    th_t = self.likelihood(data)
                    print(th_t)

                    # get plus value
                    ptm[i,j] += h0
                    ptm[i,i] -= h0
                    self.eta[1][n] = ptm_2_eta(ptm)
                    th_tp1 = self.likelihood(data)
                    print(th_tp1)

                    # get minus value
                    ptm[i,j] -= 2*h0
                    ptm[i,i] += 2*h0
                    self.eta[1][n] = ptm_2_eta(ptm)
                    th_tm1 = self.likelihood(data)
                    print(th_tm1)

                    # return theta
                    ptm[i,j] += h0
                    ptm[i,i] -= h0
                    self.eta[1][n] = ptm_2_eta(ptm)

                    # get estimate
                    I_th = (2*th_t - th_tm1 - th_tp1)/(h0**2)
                    V_th = 1.0/I_th
                    V_gamma_fine[n][i,j] = V_th
                    print('')


        SEs['Gamma_fine'] = []
        for i,x in enumerate(V_gamma_fine):
            SEs['Gamma_fine'].append(np.sqrt(x))

        self.SEs = SEs

        return SEs


    def save(self,file):

        with open(file, 'wb') as f:
            pickle.dump(self, f)

        return


    def load(self,file):

        with open(file, 'rb') as f:
            hhmm = pickle.load(f)

        return hhmm
