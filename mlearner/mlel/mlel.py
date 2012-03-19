from settings import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt    # for optional <=2D plots

"""
Joseph Edwards - CS 429 UNM Sp2012
My maximum likelihood estimate learner for the mean and covariance matrix of a Guassian
distribution in arbitrary dimensional space.
"""

class MLEControl:
    def __init__(self, sett):
        self.sett = sett

        # Default settings for DT control in settings.py
        if 'verb' not in sett:
            self.sett['verb'] = DEF_DT_VERB
        if 'plot' not in sett:
            self.sett['plot'] = DEF_MLE_PLOT
        if 'errs' not in sett:
            self.sett['errs'] = DEF_MLE_ERRS

    def reset(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def run_mlel(self, dataloader):
        Xdat, Ydat, Wdat = dataloader.get_XYW()
        D_tst, D_trn = 
        X = np.array(get_val_mat(Xdat), dtype='float')
        N = len(Xdat[0])
        Mu = X.mean(axis=0)
#        print Mu
        Sigma = np.cov(X.T)
#        print Sigma
        Sig_eig = np.linalg.eig(Sigma)

        # Report - 1st 5 components of Sigma if dim sigma vector > 5
        print "\n========= REPORT: MAXIMUM LIKELIHOOD ESTIMATOR ========="
        print "Mean (Mu) vector (<6 elts): ", Mu[:5]
        print "Eigenvals of covariant matrix (Sigma): ", Sig_eig[0]

        # Plot data? for dim Xdat row vector := N <= 2
        if N <= 2:
            self._plot_data(X, Mu)

    def _plot_data(self, X, Mu):
        plt.figure(1)
        plt.hold(True)
#        for i in range(len(self.dist_d.values()[0])):
#            plt.scatter(self.sigma_l, [self.dist_d[sigma][i] for sigma in self.sigma_l], c='b', marker='o')
        plt.scatter(X[:,0], X[:,1], c='b')
        plt.scatter(Mu[0], Mu[1], c='r', marker='o')
#        plt.plot(Sigma)
#        plt.plot(Mu)
        plt.show()
