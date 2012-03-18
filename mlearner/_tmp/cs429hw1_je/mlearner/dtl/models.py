# Simplistic decision tree

import math
import random

class LeafNode:
    def __init__(self, outcome):
        self.outcome = outcome
        self.is_leaf = True

    def __str__(self):
        return "{} ({}/{})".format(self.outcome['value'], 
                                   self.outcome['count'], 
                                   self.outcome['tab_len']) 

class DecisionNode:
    def __init__(self, next_best):
        self.attr = next_best['attr']
        self.gain = next_best['gain']
        self.tree = {}
        self.is_leaf = False

    def __str__(self):
        return self.attr

class DTree:
    """
    Class to encapsulate feature & hypothesis spaces.
    Includes method 'build_dt()' to return a decision tree
    given X (data) and Y (hypothesis vector) are set.
    """
    def __init__(self, verbosity, gain_lambda, max_iter_verb):
        self.verb = verbosity
        self.is_first_time = True
        self.iterations = 0
        self.lnode_n = 0
        self.dnode_n = 0
        self.gain_lambda = gain_lambda
        self.max_iter_verb = max_iter_verb

    # Accessors
    def get_attr_list(self, Xdata):
        return [k for k in Xdata[0]]

    def get_attr_col(self, Xdata, attr):
        return [record[attr] for record in Xdata]

    def get_col_val_set_count(self, attr, col):
        val_count = dict([(val, col.count(val)) for val in list(set(col))])
        if self.verb > 2:
            print "COUNTED VALS for ATTR=={:s}: ".format(str(attr)), val_count
        return val_count

    def get_rows_by_attr_val(self, data, attr, val):
        attr_col = self.get_attr_col(data, attr)
        match_ixl = [ix for ix in range(len(attr_col)) if attr_col[ix] == val]
        if self.verb > 2:
            print "MATCHED ROWS for ATTR=={:s}, VAL=={:s}: ".format(str(attr), str(val)), match_ixl
        return match_ixl

    def get_majority(self, Ylabels):
        val_count = self.get_col_val_set_count('Y', Ylabels)
        vc_max = max(v for k,v in val_count.iteritems())
        vc_rats = float(vc_max)/float(len(Ylabels))
        major_l = [{'value':k, 'count':v, 'tab_len':len(Ylabels), 'ratio':vc_rats} for k,v in val_count.iteritems() if v==vc_max]
        if self.verb > 2:
            print "Has target VAL counts :", val_count
            print "Majority set of Y: ", major_l
            print "Ratio to subtable size :", vc_rats
        return major_l

    def get_leaf(self, Xdata, Ylabels, attr_list):
        if not Xdata or len(attr_list) <= 0:
            major = random.choice(self.get_majority(Ylabels))
            ret_leaf = LeafNode(major) 
        elif Ylabels.count(Ylabels[0]) == len(Ylabels):
            pure = {'value':Ylabels[0], 'count':len(Ylabels), 'tab_len':len(Ylabels), 'ratio':1.0}
            ret_leaf = LeafNode(pure)
        else:
            ret_leaf = None
        if ret_leaf and self.verb > 1 and self.iterations < self.max_iter_verb:
            print "\n================\n>>> LEAF NODE: ", ret_leaf.outcome
            print "================"

        return ret_leaf

    def get_entropy(self, Ylabels, attr):
        vc_Y = self.get_col_val_set_count('Y', Ylabels)
        ent_l = []
        prob_d = {}
        for k,v in vc_Y.iteritems():
            prob = float(v)/float(len(Ylabels))
            ent_l.append(float(-prob * math.log(prob, 2)))
            prob_d[k] = prob
        entropy = sum(ent_l)
        return entropy, prob_d

    def get_gain(self, Xdata, Ylabels, attr):
        attr_col = self.get_attr_col(Xdata, attr)
        val_count = self.get_col_val_set_count(attr, attr_col)
        parent_entropy, parent_prob = self.get_entropy(Ylabels, attr)
        sub_ent_l = []
        for k,v in val_count.iteritems():
            row_ix = self.get_rows_by_attr_val(Xdata, attr, k)
            sub_labels = [Ylabels[ix] for ix in row_ix]
            weight = float(float(v)/float(len(Ylabels))) 
            sub_entropy, sub_prob = self.get_entropy(sub_labels, k)
            sub_ent_l.append((sub_prob, sub_entropy, weight))
        rows_entropy = sum([w*e for (p,e,w) in sub_ent_l])
        gain = parent_entropy - rows_entropy
        if self.verb > 1 and self.iterations < self.max_iter_verb:
            print "\nFor ATTRIBUTE {} ...".format(attr)
            print "{} records with val:count pairs: ".format(len(Ylabels)), val_count
            print "PARENT PROBs: ", parent_prob
            print "PARENT ENTROPY: ", parent_entropy
            print "GAIN: ", gain
            if self.verb > 2:
                print "\tSUB PROB: ", [p for (p,e,w) in sub_ent_l]
                print "\tENTROPY (per val): ", [e for (p,e,w) in sub_ent_l]
                print "\tWEIGHTS: ", [w for (p,e,w) in sub_ent_l]
                print "\tSUB ENTROPY (weighted): ", rows_entropy
        return gain

    def get_best_attr(self, Xdata, Ylabels, attr_list):
        gain_d = {}
        for attr in attr_list:
            gain_d[attr] = self.get_gain(Xdata, Ylabels, attr)
        gain_max = max(gain_d.values())
        gain_max_attr_l = [k for k,v in gain_d.iteritems() if v==gain_max]
        return {'attr':random.choice(gain_max_attr_l), 'gain':gain_max}

    def build_dt(self, data, labels, attr_list):
        """
        Recursive method to build decision tree
        """
        if self.is_first_time:
            self.target = str(labels.keys()[0])
            labels = labels[self.target][:]
            self.top_length = float(len(labels[:]))
            self.top_val_set_count = self.get_col_val_set_count(self.target, labels)
            self.top_entropy = self.get_entropy(labels[:], 'top')
            print "\n========================================================="
            print "BUILDING DECISION TREE with TARGET ATTRIBUTE {}".format(self.target)
            print "TOP TABLE: {} records".format(self.top_length)
            print "\tFound TARGET LABELS: ", self.top_val_set_count
            print "\tTOP ENTROPY: ", self.top_entropy
            self.is_first_time = False

        # Copy data ea. iter. out of habit
        Xdata = data[:]
        Ylabels = labels[:]
        self.iterations += 1

        if self.verb > 2 and self.iterations < self.max_iter_verb:
            print "\n***** BUILD_DT ITERATION {} *****".format(self.iterations)
            print "ATTRIBUTES list: ", attr_list

        leaf = self.get_leaf(Xdata, Ylabels, attr_list)
        if leaf:
            self.lnode_n += 1
            return leaf
        else:
            next_best = self.get_best_attr(Xdata, Ylabels, attr_list)
            if next_best['gain'] < self.gain_lambda:
                major = random.choice(self.get_majority(Ylabels))
                self.lnode_n += 1
                return LeafNode(major)
            dnode = DecisionNode(next_best)
            if self.verb > 1 and self.iterations < self.max_iter_verb:
                print "\n==================\n>>> DECISION NODE: ", next_best
                print "=================="
            for val in self.get_col_val_set_count(next_best['attr'], self.get_attr_col(Xdata, next_best['attr'])):
                row_ix = self.get_rows_by_attr_val(Xdata, next_best['attr'], val)
                split_X = [Xdata[ix] for ix in row_ix]
                split_Y = [Ylabels[ix] for ix in row_ix]
                next_attr_list = [attr for attr in attr_list if attr != next_best['attr']]
                branch = self.build_dt( split_X, split_Y, next_attr_list )
                dnode.tree[val] = branch
                self.dnode_n += 1
        return dnode
