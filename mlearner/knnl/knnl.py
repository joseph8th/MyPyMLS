from settings import *
from functions import *
import numpy as np
from scipy.spatial import distance

# K-Nearest Neighbor Learner

class KNNControl:
    def __init__(self, sett):
        self.sett = sett

        # Default settings for DT control in settings.py
        if 'verb' not in sett:
            self.sett['verb'] = DEF_DT_VERB
        if 'knn_k' not in sett:
            self.sett['knn_k'] = DEF_KNN_K
        if 'metric' not in sett:
            self.sett['metric'] = DEF_KNN_METRIC

        # Instantiate a KNNLearner ...
        self.KNN = KNNLearner()

    def reset(self):
        pass

    def train(self, D, target):
        dist = self._get_dist()
        X, Y, W = ungroup_data(D, target)
        Xmat = get_val_mat(X)
        if self.sett['metric'] != 3:
            Xmat = get_numeric_mat(Xmat)
        classes = set(Y.values()[0])

        Dmat = []
        for m in range(len(Xmat)):
            label_rec = Y.values()[0]
            d_rec = [{dist(Xmat[m], Xmat[ix]) : label_rec[ix]} for ix in range(len(Xmat)) if m != ix]
            d_min = min(d_rec)
            Dmat.append(d_min)
        
#        label_ct = [{Dmat.count(label) : label} for label in classes]
#        guess = max(label_ct)
        return Dmat


    def test(self, model, D_tst, D_trn, target):
        X_tst, Y_tst, W_tst = ungroup_data(D_tst, target)
        X_trn, Y_trn, W_trn = ungroup_data(D_trn, target)
        truth_l = []
        for ix in range(len(model)):
            truth_l.append(model[ix].values()[0] == Y_trn.values()[0][ix])
        good_g = truth_l.count(True)
        bad_g = truth_l.count(False)
        hit_rate = float(good_g)/float(len(model))
        err_rate = float(bad_g)/float(len(model))
        print "\nCORRECT GUESSES: {} out of {} = {}".format(good_g, len(model), hit_rate)
        print "\nINCORRECT GUESSES: {} out of {} = {}".format(bad_g, len(model), err_rate)

    # Basic command-line run method
    def run_knnl(self, dataloader):
        X, Y, W = dataloader.get_XYW()
        D = group_data(X,Y,W)
        D_tst, D_trn = split_nfold(D, Y.keys()[0], 2)
        if self.sett['metric'] != 0:
            trn_model = self.train(D_trn, Y.keys()[0])
            results = self.test(trn_model, D_trn, D_trn, Y.keys()[0])
        else:
            trn_model = []
            for ix in range(1,3):
                self.sett['metric'] = ix
                trn_model.append( {'metric': m, 'trn_model': self.train(D, Y.keys()[0])} ) 
        
    # Categorical metric method
    def _string_metric(self, u, v):
        truth_l = [u[ix] == v[ix] for ix in range(len(u))]
        d = 1 - (float(sum(map(int, truth_l))) / float(len(u))) 
        return d

    # Return the chosen metric
    def _get_dist(self):
        if self.sett['metric'] == 1:
            return distance.euclidean
        elif self.sett['metric'] == 2:
            return distance.cityblock
        elif self.sett['metric'] == 3:
            return self._string_metric


class KNNLearner:
#    def __init__(self, distance_metric):
#        self.dist = distance_metric

    def get_knn(self, D_trn, target):
        pass

    def test(self, model, D_tst, D_trn, target):
        pass
