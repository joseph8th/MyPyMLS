from settings import *
from functions import *
import heapq
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


    def train(self, Xdat):
        R = len(Xdat)
        X = np.array(get_val_mat(Xdat), dtype='float')

        Mu = X.mean(axis=0)
        Mu_mat = np.tile(Mu, (R,1))
        X_tmp = X - Mu_mat

        Sigma = np.cov(X_tmp.T)
        Sig_eig = np.linalg.eig(Sigma)

        return (Mu,Sigma,Sig_eig)


    def test(self, x_tst, model, target):
        Mu, Sigma, Sig_eig = model
        x = np.array(x_tst, dtype='float')
        mu = x.mean()
        sig = np.cov(x.T)

#        print x
#        print target
#        print mu
#        print Mu
#        print sig
#        print np.linalg.inv(Sigma)
#        print np.dot(np.dot(x, np.linalg.inv(Sigma)), x.T)
#        print np.dot(X_tmp, np.dot(Sigma, X_tmp.T))

        return model


    def run_mlel(self, dataloader):
        Xdat, Ydat, Wdat = dataloader.get_XYW()
        D = group_data(Xdat, Ydat, Wdat)
        N = len(Xdat[0])
        R = float(len(Xdat))
        target = Ydat.keys()[0]
        X = np.array(get_val_mat(Xdat), dtype='float')
        Y = np.array(Ydat.values()[0], dtype='float')

        class_l = list(set(Ydat.values()[0]))
        class_lixl = [get_rows_by_attr_val(D, target, y) for y in class_l]
        Xdat_cls = {}
        for i in range(len(class_l)):
            Xdat_cls[class_l[i]] = {}
            X_tmp = [Xdat[k] for k in class_lixl[i]]
            Xdat_cls[class_l[i]]['X'] = np.array(get_val_mat(X_tmp), dtype='float')
            Mu, Sigma, Sig_eig = self.train(X_tmp)
            Xdat_cls[class_l[i]]['model'] = (Mu, Sigma, Sig_eig)
            Xdat_cls[class_l[i]]['prob'] = float(len(X_tmp))/R

            # Report - 1st 5 components of Sigma if dim sigma vector > 5
            print "\n========= REPORT: MAXIMUM LIKELIHOOD ESTIMATOR ========="
            print "Class: ", class_l[i]
            print "Mean (Mu) vector (<6 elts): ", Mu[:5]
            print "Eigenvals of covariant matrix (Sigma) (<6 big):\n", heapq.nlargest(5, Sig_eig[0])

        print [Xdat_cls[k]['prob']*Xdat_cls[k]['model'][0] for k in Xdat_cls.iterkeys()]
            
        Mu, Sigma, Sig_eig = self.train(Xdat)
#        print X
#        print np.sum(np.dot(np.linalg.inv(Sigma), X.T).T, axis=0)/R
#        print "\n"
        results = self.test(X[0], (Mu, Sigma, Sig_eig), Y[0])
#        print results

        # Optional compute error rates
        if self.sett['errs']:
            D = group_data(Xdat, Ydat, Wdat)

            # Empirical error rate by 10fold cross-validation
            results_l = []
            for _ in range(10):
                D_tst, D_trn = split_nfold(D, target, 10)
                X_trn, Y_trn, W_trn = ungroup_data(D_trn, target)
                mle_model = self.train(X_trn)
                results = self.test(D_tst, mle_model, target)
            print results

        # Plot data? for dim( Xdat row ) := N <= 2
        if N <= 2 and self.sett['plot']:
            self._plot_data(np.array(get_val_mat(Xdat), dtype='float'), Mu)


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
