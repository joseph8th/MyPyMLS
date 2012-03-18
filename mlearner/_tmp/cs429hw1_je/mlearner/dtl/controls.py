from settings import *
from string import *
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
            sett['verb'] = DEF_DT_VERB
        if 'gain_lambda' not in sett:
            sett['gain_lambda'] = DEF_DT_GAIN_LAMBDA
        if 'max_iter_verb' not in sett:
            sett['max_iter_verb'] = DEF_DT_MAX_ITER_VERB

    # Split strings by whitespace or comma
    def _split_string(self, data):
        if data[0].find(',') != -1:
            data = map(lambda c: c.split(','), data)
        else:
            data = map(lambda c: c.split(), data)
        if self.sett['verb'] < 0: 
            print "\nSplit data:\n==================\n", data
        return data

    # Cleanup and format data table for DTree
    def _format_records(self, data):
        dl = [] 
        label_row = map(lambda c: c.strip(), data[0])
        for record in data[1:]:
            record = map(lambda c: c.strip(), record)
            dl.append(dict([ (label_row[col], record[col]) for col in range(len(record)) ]))
        if self.sett['verb'] < 0:
            print "\nFormatted records:\n=================\n", dl
        return dl

    # Load file from disk
    def _load_data(self, fname):
        f = open(DATA_PATH + fname)
        D = f.readlines()
        f.close()
        if self.sett['verb'] < 0:
            print "\nLoaded {}:\n=================\n".format(fname), D
        return D

    # Load and format data table and label vector from separate '.dat' files
    # NOTE: change variables 'labels' here to 'names' or something ...
    def _load_dat(self, (xfile, yfile), has_labels):
        print "\nLoading data (X) from {}, target (Y) from {}.".format(xfile, yfile)
        X = self._split_string(self._load_data(xfile))
        Y = self._split_string(self._load_data(yfile))
        Y = [r[0] for r in Y]
        # If no attr names, make some up and forget about it ...
        if not has_labels:
            label_row = ["A{}".format(col) for col in range(len(X[0]))]
            X.insert(0, label_row)
            Y.insert(0, 'Y')
        X = self._format_records(X)
        Y = {Y[0]: Y[1:]}
        return (X,Y)

    def _load_arff(self, dfile):
        print "\nLoading data from {}.".format(dfile)
        D = self._load_data(dfile)
        meta_l = self._split_string([l for l in D if l.startswith('@')])
        target = [r[1] for r in meta_l if r[0] == '@relation'][0]
        label_row = [a[1] for a in meta_l if a[0] == '@attribute']
        strip_D = map(strip, D)
        dix = [strip_D.index(x) for x in strip_D if x.startswith('@data')][0]
        strip_D = [strip_D[i] for i in range(dix + 1, len(strip_D)) if not strip_D[i].startswith('%')]
        strip_D = self._split_string(strip_D)
        strip_D.insert(0, label_row)
        strip_D = self._format_records(strip_D)
        Y = [strip_D[x][target] for x in range(len(strip_D))]
        Y = {target: Y[:]}
        for record in strip_D:
            del record[target]
        X = strip_D[:]
        return (X,Y)

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


    # Where the action happens
    def run_dtl(self):
        """
        Main Decision Tree view/controller method for mlearner.py.
        """
        if self.sett['ftype'] == 'dat':
            X,Y = self._load_dat(self.sett['ftuple'], self.sett['labels'])
        elif self.sett['ftype'] == 'arff':
            X,Y = self._load_arff(self.sett['ftuple'])
        if X and Y:
            DT = DTree(self.sett['verb'], self.sett['gain_lambda'], self.sett['max_iter_verb'])
            full_attr_list = DT.get_attr_list(X)
            tree = DT.build_dt(X, Y, full_attr_list)

            # Output results based on verbosity: more < 0 = I/O, more > 0 = DTL, 0 = def
            if self.sett['verb'] < -1:
                self._print_input(X, Y, full_attr_list)
            if self.sett['verb'] > -1:
                if self.sett['gain_lambda'] == 0.0:
                    print "\n===============\nUnpruned Tree:\n"
                else:
                    print "\n===============\nPruned Tree (by gain < {}:\n".format(self.sett['gain_lambda'])
                if self.sett['verb'] > 0:
                    self._print_tree(tree, 5, ["",])
                print "\n{} total nodes\n{} leaf nodes".format(DT.lnode_n + DT.dnode_n, DT.lnode_n)
