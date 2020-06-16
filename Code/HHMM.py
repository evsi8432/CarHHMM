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
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

from copy import deepcopy

import divebomb

import time
import pickle


class HHMM:


    def __init__(self,pars,data):

        self.pars = pars
        self.theta = []
        self.eta = []
        self.ptm = []
        self.data = data

        eta_crude = -1.0 + 0.1*np.random.normal(size=(self.pars.K[0],self.pars.K[0]))
        ptm_crude = np.exp(eta_crude)
        ptm_crude = (ptm_crude.T/np.sum(ptm_crude,1)).T

        self.eta.append(eta_crude)
        self.ptm.append(ptm_crude)

        eta_fine = []
        ptm_fine = []
        for _ in range(self.pars.K[0]):
            eta_fine_k = -1.0 + 0.1*np.random.normal(size=(self.pars.K[1],self.pars.K[1]))
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
                quantiles = np.linspace(1/(2*K),1-(1/(2*K)),K)

                # first find mu
                if feature in ['Ax','Ay','Az']:
                    theta[0][feature]['mu'] = np.mean(feature_data)*np.ones(K)
                else:
                    theta[0][feature]['mu'] = np.quantile(feature_data,quantiles)

                # then get varaince of each quantile set of data
                data_sorted = np.sort(feature_data)
                n = len(data_sorted)
                for k0 in range(K):
                    if settings['f'] != 'vonmises':
                        std = np.std(data_sorted[int(k0*n/K):int((k0+1)*n/K)])
                        theta[0][feature]['sig'][k0] = max(0.1,std)
                    else:
                        theta[0][feature]['sig'][k0] = 1.0

                # finally update correlations randomly
                theta[0][feature]['corr'] = np.random.random(size=K) - 2.0



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
                    quantiles = np.linspace(1/(2*K),1-(1/(2*K)),K)

                    # first find mu
                    if feature in ['Ax','Ay','Az']:
                        theta[1][k0][feature]['mu'] = np.mean(feature_data)*np.ones(K)
                    else:
                        theta[1][k0][feature]['mu'] = np.quantile(feature_data,quantiles)
                    theta[1][k0][feature]['mu'] *= norm.rvs(1,0.01)

                    # then get varaince of each quantile set of data
                    data_sorted = np.sort(feature_data)
                    n = len(data_sorted)
                    for k1 in range(K):
                        if settings['f'] != 'vonmises':
                            std = np.std(data_sorted[int(k1*n/K):int((k1+1)*n/K)])
                            theta[1][k0][feature]['sig'][k1] = max(0.1,std)
                        else:
                            theta[1][k0][feature]['sig'][k1] = 1.0
                        theta[1][k0][feature]['sig'][k1]*= norm.rvs(1,0.01)

                    # finally update correlations randomly
                    theta[1][k0][feature]['corr'] = 0.1*np.random.random(size=K) - 1.0

        self.theta = theta

        return


    def initalize_theta_sim(self):



        return


    def find_log_p_yt_given_xt(self,level,feature,data,data_tm1,mu,sig,corr,sample=0):

        # find log density of feature
        if self.pars.features[level][feature]['f'] == 'multivariate_normal':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]

            if sample > 0:
                return multivariate_normal.rvs(mu,sig,sample)
            else:
                return multivariate_normal.logpdf(mu,data[feature],sig)

        # find log density of feature
        if self.pars.features[level][feature]['f'] == 'normal':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]

            if sample > 0:
                return norm.rvs(mu,sig,sample)
            else:
                return norm.logpdf(data[feature],mu,sig)

        elif self.pars.features[level][feature]['f'] == 'gamma':

            # find new mean if there is autocorrelation
            if self.pars.features[level][feature]['corr'] and data_tm1 is not None:
                mu = (1.0-corr)*mu + corr*data_tm1[feature]

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
            else:
                pass

            kappa = sig

            if sample > 0:
                return vonmises.rvs(kappa,loc=mu,size=sample)
            else:
                return vonmises.logpdf(data[feature],kappa,loc=mu)


    def dive_likelihood(self,dive_data,state):

        # find tpm
        self.eta[1][state][np.diag_indices(self.pars.K[1])] = 0
        ptm = np.exp(self.eta[1][state])
        ptm = (ptm.T/np.sum(ptm,1)).T
        #ptm = self.ptm[1][state]
        log_ptm = np.log(ptm)

        # find the initial distribution (stationary distribution)
        delta = np.ones((1,self.pars.K[1]))/self.pars.K[1]
        for _ in range(10):
            delta = delta.dot(ptm)
        log_phi = np.log(delta)

        # initialize values
        log_L = 0
        seg_tm1 = dive_data[0]

        # iterate through dive segments
        for i,seg in enumerate(dive_data):

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
        #ptm = self.ptm[0]
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


    def train_DM(self,data,max_iters=10,tol=0.01,eps=10e-6):

        options = {'maxiter':10,'disp':False}
        prev_l = self.likelihood(data)

        for _ in range(max_iters):

            print(prev_l)

            ### start with crude eta ###
            def loss_fn(x):
                ind = 0

                # update crude eta
                self.eta[0] = x[0:self.pars.K[0]**2].reshape((self.pars.K[0],
                                                              self.pars.K[0]))
                l = -self.likelihood(data)
                return l

            # define inital value
            x0 = []
            x0.extend(self.eta[0].flatten())

            # optimize
            res = minimize(loss_fn, x0, method='Nelder-Mead',options=options)

            # update final values
            x = np.copy(res['x'])
            self.eta[0] = x.reshape((self.pars.K[0],self.pars.K[0]))

            #### then do crude theta ###
            for k0 in range(self.pars.K[0]):
                for feature in self.pars.features[0]:

                    def loss_fn(x):
                        ind = 0
                        if feature in ['Ay','Az']:
                            self.theta[0][feature]['mu'][k0] = x[ind]
                            self.theta[0][feature]['sig'][k0] = self.theta[0]['Ax']['sig'][k0]
                            self.theta[0][feature]['corr'][k0] = self.theta[0]['Ax']['corr'][k0]
                            ind += 1
                        else:
                            self.theta[0][feature]['mu'][k0] = x[ind]
                            self.theta[0][feature]['sig'][k0] = max(x[ind+1],eps)
                            self.theta[0][feature]['corr'][k0] = x[ind+2]
                            ind += 3

                        return -self.likelihood(data)

                    # define inital value
                    x0 = []
                    x0.append(self.theta[0][feature]['mu'][k0])
                    if feature not in ['Ay','Az']:
                        x0.append(self.theta[0][feature]['sig'][k0])
                        x0.append(self.theta[0][feature]['corr'][k0])

                    # optimize
                    res = minimize(loss_fn, x0, method='Nelder-Mead',options=options)

                    # update final values
                    x = np.copy(res['x'])
                    if feature in ['Ay','Az']:
                        self.theta[0][feature]['mu'][k0] = x[0]
                        self.theta[0][feature]['sig'][k0] = self.theta[0]['Ax']['sig'][k0]
                        self.theta[0][feature]['corr'][k0] = self.theta[0]['Ax']['corr'][k0]
                    else:
                        self.theta[0][feature]['mu'][k0] = x[0]
                        self.theta[0][feature]['sig'][k0] = max(x[1],eps)
                        self.theta[0][feature]['corr'][k0] = x[2]

            ### then do fine eta ###
            for k0 in range(self.pars.K[0]):

                def loss_fn(x):
                    self.eta[1][k0] = x.reshape((self.pars.K[1],self.pars.K[1]))
                    l = -self.likelihood(data)
                    return l

                # define inital value
                x0 = []
                x0.extend(self.eta[1][k0].flatten())

                # optimize
                res = minimize(loss_fn, x0, method='Nelder-Mead',options=options)

                # update final values
                x = np.copy(res['x'])
                self.eta[1][k0] = x.reshape((self.pars.K[1],self.pars.K[1]))


            ### finally do fine theta ###
            for k1 in range(self.pars.K[1]):

                if self.pars.share_fine_states:
                    K0 = 1
                else:
                    K0 = self.pars.K[0]

                for feature in self.pars.features[1]:

                    for k0 in range(K0):

                        def loss_fn(x):
                            ind = 0
                            for k00 in range(self.pars.K[0]):
                                if self.pars.share_fine_states or (k00 == k0):
                                    if feature in ['Ay','Az']:
                                        self.theta[1][k00][feature]['mu'][k1] = x[ind]
                                        self.theta[1][k00][feature]['sig'][k1] = self.theta[1][k0]['Ax']['sig'][k1]
                                        self.theta[1][k00][feature]['corr'][k1] = self.theta[1][k0]['Ax']['corr'][k1]
                                        if not self.pars.share_fine_states or (k00 == self.pars.K[0]-1):
                                            ind += 1
                                    else:
                                        self.theta[1][k00][feature]['mu'][k1] = x[ind]
                                        self.theta[1][k00][feature]['sig'][k1] = max(x[ind+1],eps)
                                        self.theta[1][k00][feature]['corr'][k1] = x[ind+2]
                                        if not self.pars.share_fine_states or (k00 == self.pars.K[0]-1):
                                            ind += 3

                            l = -self.likelihood(data)
                            return l

                        # define inital value
                        x0 = []
                        x0.append(self.theta[1][k0][feature]['mu'][k1])
                        if feature not in ['Ay','Az']:
                            x0.append(self.theta[1][k0][feature]['sig'][k1])
                            x0.append(self.theta[1][k0][feature]['corr'][k1])


                        # optimize
                        if (not self.pars.share_fine_states) or (k0 == 0):
                            res = minimize(loss_fn, x0, method='Nelder-Mead',options=options)

                        # update final values
                        x = np.copy(res['x'])
                        ind = 0
                        for k00 in range(self.pars.K[0]):
                            if self.pars.share_fine_states or (k00 == k0):
                                if feature in ['Ay','Az']:
                                    self.theta[1][k00][feature]['mu'][k1] = x[ind]
                                    self.theta[1][k00][feature]['sig'][k1] = self.theta[1][k0]['Ax']['sig'][k1]
                                    self.theta[1][k00][feature]['corr'][k1] = self.theta[1][k0]['Ax']['corr'][k1]
                                    if not self.pars.share_fine_states or (k00 == self.pars.K[0]-1):
                                        ind += 1
                                else:
                                    self.theta[1][k00][feature]['mu'][k1] = x[ind]
                                    self.theta[1][k00][feature]['sig'][k1] = max(x[ind+1],eps)
                                    self.theta[1][k00][feature]['corr'][k1] = x[ind+2]
                                    if not self.pars.share_fine_states or (k00 == self.pars.K[0]-1):
                                        ind += 3

            curr_l = self.likelihood(data)
            if abs(curr_l - prev_l) < tol:
                break
            else:
                prev_l = curr_l

        return (self.theta,self.eta)


    def train_EM(self,data,tol,max_iters):

        # train using the Baum-Welch algorithm

        for _ in range(max_iters):

            # crude E step
            _,_,gamma0,xi0 = self.fwd_bwd(data)

            # fine E step
            gamma1 = []
            xi1 = []
            for t0,dive in enumerate(data):

                subdive_data = dive['subdive_features']
                gamma1t0 = np.zeros((self.pars.K[0],self.pars.K[1],len(subdive_data)-1))
                xi1t0 = np.zeros((self.pars.K[0],self.pars.K[1],self.pars.K[1],len(subdive_data)-1))

                for k0 in range(self.pars.K[0]):

                    _,_,gamma1k,xi1k = self.fwd_bwd(subdive_data,[1,k0])
                    gamma1t0[k0,:,:] = gamma0[k0,t0]*gamma1k
                    xi1t0[k0,:,:,:] = gamma0[k0,t0]*xi1k

                gamma1.append(gamma1t0)
                xi1.append(xi1t0)

            gamma1 = np.concatenate(gamma1,axis=1)
            xi1 = np.concatenate(xi1,axis=2)

            # crude M step
            # self.ptm[0] = np.sum(xi0,axis=2)/np.sum(gamma0,axis=1)

            # estimate normal probs


            # fine M step

        return


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


    def save(self,file):

        with open(file, 'wb') as f:
            pickle.dump(self, f)

        return


    def load(self,file):

        with open(file, 'rb') as f:
            hhmm = pickle.load(f)

        return hhmm
