import sys
import random

### Global error handler
def print_err(msg):
    print >> sys.stderr, msg


### On-the-fly data mutators
def group_data(X, Y, W):
    D = [X[ix].copy() for ix in range(len(X))]

    for ix in range(len(X)):
        D[ix][Y.keys()[0]] = Y.values()[0][ix]
        D[ix]['weight'] = W[ix]

    return D

def ungroup_data(D, target):
    X = [D[ix].copy() for ix in range(len(D))]
    Y = {target : [X[ix].pop(target) for ix in range(len(X))]}
    W = [X[ix].pop('weight') for ix in range(len(X))]
    return (X, Y, W)


# The n-fold split...
def split_nfold(data, target, n):
    D = [data[ix].copy() for ix in range(len(data))]
    N = len(D)/n
    random.shuffle(D)
    D_test = D[:N]
    D_train = D[N:]
    return D_test, D_train

### Arbitrary type accessors and mutators


def get_attr_list(Xdata):
    return [k for k in Xdata[0]]

def get_attr_col(Xdata, attr):
    return [record[attr] for record in Xdata]

def get_col_val_set_count(attr, col, W):
    val_count = dict([(val, col.count(val)) for val in list(set(col))])
    wval_count = {}

    for val, count in val_count.iteritems():
        wval_l = [W[ix] for ix in range(len(col)) if col[ix] == val]
        wval_count[val] = sum(wval_l)

    return wval_count

def get_rows_by_attr_val(data, attr, val):
    attr_col = get_attr_col(data, attr)
    match_ixl = [ix for ix in range(len(attr_col)) if attr_col[ix] == val]
    return match_ixl

def get_entropy(Ylabels, W, attr):
    vc_Y = get_col_val_set_count('Y', Ylabels, W)
    ent_l = []
    prob_d = {}

    for val, count in vc_Y.iteritems():
        prob = float(count)/float(sum(W))
        ent_l.append(float(-prob * math.log(prob, 2)))
        prob_d[val] = prob

    entropy = sum(ent_l)
    return entropy, prob_d

def get_gain(Xdata, Ylabels, W, attr):
    attr_col = get_attr_col(Xdata, attr)
    val_count = get_col_val_set_count(attr, attr_col, W)
    parent_entropy, parent_prob = get_entropy(Ylabels, W, attr)

    sub_ent_l = []
    for val, count in val_count.iteritems():
        row_ix = self.get_rows_by_attr_val(Xdata, attr, val)
        sub_labels = [Ylabels[ix] for ix in row_ix]
        sub_weights = [W[ix] for ix in row_ix]
        prob = float(count)/float(sum(W)) 
        sub_entropy, sub_prob = get_entropy(sub_labels, sub_weights, val)
        sub_ent_l.append((sub_prob, sub_entropy, prob))

    rows_entropy = sum([p*se for (sp,se,p) in sub_ent_l])
    gain = parent_entropy - rows_entropy
    return gain
    
def get_val_mat(X):
    Xmat = []
    for m in range(len(X)):
        rec_l = [val for val in X[m].itervalues()]
        Xmat.append(rec_l)
    return Xmat

def get_numeric_mat(Xmat):
    Xnum = []
    for rec in Xmat:
        Xnum.append(map(float, rec))
    return Xnum
