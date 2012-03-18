from functions import *
import numpy as np
import math

"""
AdaBoost controller class
"""
class AdaControl:

    # Convert binary-valued labels to {-1,1}
    def _set_ada_class(self, Y):
        label_set = list(set(Y.values()[0]))
        if not len(label_set) == 2:
            print "AdaBoost requires binary class set."
            exit()
        else:
            label_map = {label_set[0]: -1.0, label_set[1]: 1.0}
            return {'label_map': label_map, 'Y_ada': [label_map[y] for y in Y.values()[0]]}

    def run_ada(self, data_loader, learn_con):
        X, Y, W = data_loader.get_XYW()
        target = Y.keys()[0]
        Y_ada = self._set_ada_class(Y)
        D = group_data(X, {target: Y_ada['Y_ada']}, W)
        D_tst, D_trn = split_nfold(D, target, 2)

        boost = AdaBoost(learn_con)
        ada_model = boost.ada_train(D_trn, target, 
                                    learn_con.sett['ada_iter'], 
                                    learn_con.sett['ada_errlim'])

"""
AdaBoost model class
"""
class AdaBoost:
    def __init__(self, learn_con):
        self.learner = learn_con

    def ada_train(self, D_trn, target, iter_n, err_lim):
        np_eps = np.finfo(float).eps
        ada_model = {'weights': [], 'trn_model': []}
        X_trn, Y_trn, W_trn = ungroup_data(D_trn, target)
        # Init weights vector
        Y_t = np.array(Y_trn.values()[0])
        W_t = np.array(W_trn)/float(len(W_trn))

        for t in range(iter_n):
            self.learner.reset()

            # Group weight_t with training set
            D_trn = group_data(X_trn, Y_trn, list(W_t))

            # Train the learner on training set
            trn_model = self.learner.train(D_trn, target)
            ada_model['trn_model'].append(trn_model)

            # Test the training set on learner test
            test_results = self.learner.test(trn_model, D_trn, D_trn, target)
            guess_l = [test_results['test_list'][i]['out_guess'].outcome['value'] \
                           for i in range(len(test_results['test_list']))]
            guess_l = np.array(guess_l)

            # Get indicator then error
            indic_l = (np.ones_like(guess_l) - (guess_l*Y_t))/2
            err_eps = sum(W_t * indic_l)/sum(W_t)
            if err_eps <= err_lim:
                break

            # Get weight for this iter
            weight = math.log((1.0 - err_eps)/err_eps)
            ada_model['weights'].append(weight)

            # Set weights
            tmp = W_t * np.array([math.exp((-1.0) * weight * guess_l[ix] * Y_t[ix]) for ix in range(len(W_t))])
            W_t = tmp / sum(tmp)

            if self.learner.sett['verb'] > -1:
                print "Iter err_eps: ", err_eps
                print "Iter weight: ", weight
#                print W_t

#        weights_ary = np.array(ada_model['weights'])
#        ada_model['weights'] = weights_ary / sum(weights_ary)
        return ada_model

    def ada_test(self, ada_model, D_tst, D_trn, target):
        X_tst, Y_tst, W_tst = ungroup_data(D_tst, target)
        W_t = list(np.ones(len(W_tst)))
        D_t = group_data(X_tst, Y_tst, W_t)
#        X_trn, Y_trn, W_trn = ungroup_data(D_trn, target)

        for ix in range(len(ada_model['weights'])):
            test_results = self.learner.test(ada_model['trn_model'], D_t, D_trn, target)
            
