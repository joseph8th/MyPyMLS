from settings import *
from functions import *
from dataloader import *
from dtl.models import *

class DTControl:
    """
    A view/controller class for my basic decision tree app.
    Joseph Edwards <jee8th@unm.edu>
    CS429 - Spring 2012 - UNM
    """
    def __init__(self, sett):
        self.sett = sett

        # Default settings for DT control in settings.py
        if 'verb' not in sett:
            self.sett['verb'] = DEF_DT_VERB
        if 'gain_lambda' not in sett:
            self.sett['gain_lambda'] = DEF_DT_GAIN_LAMBDA
        if 'max_iter_verb' not in sett:
            self.sett['max_iter_verb'] = DEF_DT_MAX_ITER_VERB

        # Instantiate a DTree ...
        self.DT = DTree(self.sett['verb'], 
                        self.sett['gain_lambda'], 
                        self.sett['max_iter_verb'])


    ### A bunch of print methods to stdout

    def _print_input(self, data, labels, attr_list):
        print "\nXdata Input:\n==============\n", data
        print "\nYlabels Input:\n==============\n", labels
        print "\nAttribute List:\n==============\n", attr_list

    def _prefix(self, (ps, ch, s), depth):
        stem = rjust("", 4, ch)
        node_str = "{}{}".format(stem, s)
        pref = rjust(ps, depth-4, " ")
        return pref + node_str

    def _print_tree(self, node, depth, pre_l):
        if node.is_leaf:
            leaf_pre = self._prefix(("\\","-",">"), depth)
            print "{}{} {}".format("".join(pre_l[:len(pre_l)]), leaf_pre, node)
        else:
            for branch in node.tree:
                print "{}{} = {} ({}) ".format("".join(pre_l), 
                                               str(node), 
                                               str(branch), 
                                               str(node.gain))
                branch_pre = self._prefix(("|"," ",""), depth)
                pre_l.append(branch_pre)
                self._print_tree(node.tree[branch], depth, pre_l)
                pre_l.pop()

    def _get_report_head(self):
        head = "\n===============******* REPORT *******===============\n"
        if 'nfold' in self.sett:
            head += "{}-FOLD TEST against ".format(self.sett['nfold'])
        if self.sett['gain_lambda'] == 0.0:
            head += "Unpruned Tree on Full Set\n"
        else:
            head +="Pruned Tree on Full Set (stop-gain < {})\n".format(self.sett['gain_lambda'])
        return head


    # Generic training controller
    def train(self, D, target):
        X, Y, W = ungroup_data(D, target)
        attr_list = self.DT.get_attr_list(X)
        tree = self.DT.build_dt(X, Y, W, attr_list)

        if self.sett['verb'] > -1:
            print self._get_report_head()
            if self.sett['verb'] > 0:
                self._print_tree(tree, 5, ["",])
            print "\n{} total nodes\n{} leaf nodes\n".format(self.DT.lnode_n + self.DT.dnode_n, 
                                                             self.DT.lnode_n)
        return tree

    # Generic testing controller
    def test(self, tree, D_tst, D_trn, target):
        tst_l = []
        accu_d = {'good':[], 'bad':[]}
        X_trn, Y_trn, W_trn = ungroup_data(D_trn, target)
        X_tst, Y_tst, W_tst = ungroup_data(D_tst, target)

        if self.sett['verb'] > 1:
            print "QUERYING {} test instances...".format(len(X_tst))

        for ix in range(len(X_tst)):
            results = self.DT.query(tree, X_tst[ix], [])
            out_actual = Y_tst.values()[0][ix]
            out_guess = results[len(results)-1]

            if not hasattr(out_guess, "outcome"):
                major = random.choice(self.DT.get_majority(Y_trn.values()[0], W_trn))
                out_guess = LeafNode(major)
                results += ["THEN ", out_guess]

            is_correct = out_actual == out_guess.outcome['value']
            tst_l.append({'results' : results[:len(results)-1], 'out_guess': out_guess, 
                          'out_actual' : out_actual, 'is_correct' : is_correct})
            if is_correct:
                accu_d['good'].append(ix)
            else:
                accu_d['bad'].append(ix)

            if self.sett['verb'] > 2:
                print "TEST vector {}: ".format(ix), X_tst[ix]
                print "OUTCOME {}: ".format(ix+1), "".join(tst_l[ix]['results'])
                print "\tGuess: {}\tActual: {}\tEqual? ".format(tst_l[ix]['out_guess'],
                                                                tst_l[ix]['out_actual'],
                                                                ), tst_l[ix]['is_correct']
        # Count up correct vs/ incorrect guesses and ratios
        good_g = len(accu_d['good'])
        bad_g = len(accu_d['bad'])
        guess_ct = good_g + bad_g
        hit_rate = float(good_g)/float(guess_ct)
        err_rate = float(bad_g)/float(guess_ct)

        if self.sett['verb'] > -1:
            print "============="
            print "CORRECT GUESSES: {} out of {} = {:.06f}".format(good_g, guess_ct, hit_rate)
            print "WRONG GUESSES: {} out of {} = {:.06f}".format(bad_g, guess_ct, err_rate)
            print "=============\n"

        test_results = {'good_g':good_g, 'bad_g':bad_g, 'guess_ct':guess_ct,
                        'hit_rate':hit_rate, 'err_rate':err_rate, 'test_list': tst_l}
        return test_results

    # Controller for n-fold cross-validation
    def _run_nfold(self, data, target, n):
        accu_l = []
        # Split the tree n-fold times
        for nix in range(n):
            self.reset()
            D = data[:]
            D_tst, D_trn = split_nfold(D, target, n)
            tree = self.train(D_trn, target)

            # Query the tree for each test vector in the fold
            test_results = self.test(tree, D_tst, D_trn, target)
            accu_l.append(test_results)

            if self.sett['verb'] < -1:
                print "\nTEST SET:\n========\n", D_tst
                print "\nTRAIN SET:\n========\n", D_trn

        # Report test results stats to return as results
        avg_hit_rate = sum([p['hit_rate'] for p in accu_l])/float(n)
        avg_err_rate = sum([p['err_rate'] for p in accu_l])/float(n)

        if self.sett['verb'] > -1:
            print "\n====================\nMEAN CORRECT GUESSES = {:.02f}%".format(avg_hit_rate)
            print "\n====================\nMEAN WRONG GUESSES = {:.02f}%".format(avg_err_rate)


    def reset(self):
        self.DT.reset_build()

    """
    Command-line basic DTL view/controller method.
    """
    def run_dtl(self, dataloader):
        X,Y,W = dataloader.get_XYW()

        # Top tree on full data set
        D = group_data(X,Y,W)
        tree = self.train(D, Y.keys()[0])

        # N-fold cross-validation
        if 'nfold' in self.sett:
            results = self._run_nfold(D[:], Y.keys()[0], self.sett['nfold'])

        # Output results based on verbosity: more < 0 = I/O, more > 0 = DTL, 0 = def
        if self.sett['verb'] < -2:
            self._print_input(X, Y, self.DT.get_attr_list)
