from dtl.controls import *
from knnl.knnl import *
from mlel.mlel import *
from tools.controls import *
from tools.adaboost import *

"""
Command-line and sys-level config and control
for my basic machine learning packages.

Joseph Edwards <jee8th@unm.edu>
CS429 - Spring 2012 - UNM
"""

def _print_help():
    print "\nUSAGE: python nmlearner.py &&"
    print "\t[--dtl || --knn] -f [*.dat *.dat] [-l] || [*.arff] || [*.data *.names] &&"
    print "\t[-w [*.dat]] [-v [-3:3]] [-g [0.0:1.0]] [-i [0:]] [-n [0:99]] [-a [1:] [:3]]"
    print "\t|| [--tool 1 [mu dim n]]"
    print "\n--dtl = decision tree learner:\n\t-f, --file = filename(s) (.dat or .arff)\n\t-v, --verb = verbosity"
    print "\t-w, --weights = input weights file\n\t-g, --gain = stop gain prune\n\t-i, --iter = verb iter limit"
    print "\t-n, --nfold = n-fold split-train-test\n\t-a, --ada = ada boost iters and metric #" 
    print "\tData files must be stored in \'./data/\' folder, or set absolute DATA_PATH in settings.py."
    print "\n--tool = analysis tool:"
    print "\t1 = tool 1 (plot noisy vectors):"
    print "\t\tmu = mean for normal distr\n\t\tdim = d-dimension of vectors\n\t\tn = size of random noise pool (for average)"
    print "\t\tSigma = standard deviation list (x-axis of plot) is set by SIGMA_LIST in settings.py."

# Get file settings (required)
def _get_file_sett(sa):
    sett = {}
    sett['labels'] = False
    l = len(sa)
    i = 0
    while i < l:
        # Data file req.: X & Y .dat files, Weka .arff, or UCI .data
        if sa[i] in ['-f', '--file'] and l-i > 1:
            i += 1
            fnX = sa[i]
            if fnX.endswith(".dat") and l-i > 1:
                i += 1
                fnY = sa[i]
                if not fnY.endswith(".dat"):
                    return None
                sett['ftype'] = 'dat'
                sett['ftuple'] = (fnX, fnY)
            elif fnX.endswith(".arff"):
                sett['ftype'] = 'arff'
                sett['ftuple'] = fnX
            elif fnX.endswith(".data") and l-i > 1:
                i += 1
                fnY = sa[i]
                if not fnY.endswith(".names"):
                    return None
                sett['ftype'] = 'uci'
                sett['ftuple'] = (fnX, fnY)
            else:
                return None
        # DT does file data have label header row(s)?
        elif sa[i] in ['-l', '--label']:
            sett['labels'] = True
        i += 1
    return sett

# Get general mlearner settings
def _get_mlopt_sett(sa, sett):
    l = len(sa)
    i = 0
    while i < l:
        # AdaBoost # iters & errlim choice
        if sa[i] in ['-a', '--ada'] and l-i > 2:
            i += 1
            try:
                ada_iter = int(sa[i])
                ada_errlim = float(sa[i+1])
            except (ValueError):
                return None
            if not ada_iter <= 0 and ada_errlim < 0.5:
                sett['ada_iter'] = ada_iter
                sett['ada_errlim'] = ada_errlim
            else:
                return None
        # Weights file (not required)
        elif sa[i] in ['-w', '--weights'] and l-i > 1:
            i += 1
            wfn = sa[i]
            if not wfn.endswith(".dat"):
                return None
            sett['wfile'] = wfn
        # Verbosity level: <0 = control, >0 = model
        elif sa[i] in ['-v', '--verb'] and l-i > 1:
            i += 1
            verb_val = sa[i]
            try:
                verb_val = int(verb_val)
            except (ValueError):
                return None
            if verb_val < -3 or verb_val > 3:
                return None
            else:
                sett['verb'] = verb_val
        i += 1
    return sett

# parse, validate command line args & generate settings dict for pkg
def _get_dtl_sett(sa, sett):

    sett['pkg'] = 'dtl'
    l = len(sa)
    i = 0
    while i < l:
        # DT gain_lambda ez-prune
        if sa[i] in ['-g', '--gain'] and l-i > 1:
            i += 1
            gain_lambda = sa[i]
            try:
                gain_lambda = float(gain_lambda)
            except (ValueError):
                return None
            if gain_lambda < 0 or gain_lambda > 1:
                return None
            else:
                sett['gain_lambda'] = gain_lambda
        # DT max iterations of verbosity
        elif sa[i] in ['-i', '--iter'] and l-i > 1:
            i += 1
            max_iter_verb = sa[i]
            try:
                max_iter_verb = int(max_iter_verb)
            except (ValueError):
                return None
            if max_iter_verb < 0:
                return None
            else:
                sett['max_iter_verb'] = max_iter_verb
        # N-Fold cross-validation # folds
        elif sa[i] in ['-n', '--nfold'] and l-i > 1:
            i += 1
            try:
                nfold = int(sa[i])
            except (ValueError):
                return None
            if nfold < 0 or nfold > 99:
                return None
            else:
                sett['nfold'] = nfold            
        i += 1 # endwhile
    return sett

# Settings for KNNControl and learner    
def _get_knnl_sett(sa, sett):

    sett['pkg'] = 'knn'
    l = len(sa)
    i = 0
    while i < l:
        if sa[i] == '-k' and l-i > 1:
            i += 1
            try:
                knn_k = int(sa[i])
            except (ValueError):
                return None
            if knn_k > 0:
                sett['knn_k'] = knn_k
            else:
                return None
        elif sa[i] in ['-m', '--metric'] and l-i > 1:
            i += 1
            try:
                metric = int(sa[i])
            except (ValueError):
                return None
            if metric < 4:
                sett['metric'] = metric
            else:
                return NOne
        i += 1
    return sett


# Get settings for max likelihood estimator
def _get_mlel_sett(sa, sett):
    sett['pkg'] = 'mle'
    l = len(sa)
    i = 0
    while i < l:
        if sa[i] in ['-p', '--plot']:
            sett['plot'] = True
        if sa[i] in ['-e', '--errs']:
            sett['errs'] = True
        i += 1
    return sett


# Get settings for any aux. tools
def _get_tool_sett(sa):
    sett = {}
    sett['pkg'] = 'tool'
    l = len(sa)
    i = 0
    # Plot norms(noisy data - noisy labels) for 'dim' dimensions 
    if sa[i] == '1' and l-i > 3:
        sett['tool'] = '1'
        try:
            sett['mu'] = float(sa[i+1])
#                    sett['sigma'] = float(sa[i+2])
            sett['dim'] = int(sa[i+2])
            sett['n'] = int(sa[i+3])
        except (ValueError):
            return None
    else:
        return None

    return sett

# function to switch get on settings type
def _get_settings(args):

    if len(args) < 3:
        return None
    else:
        sa = args[1:]
        if sa[0] == '--dtl':
            sett = _get_file_sett(sa[1:])
            return _get_dtl_sett(sa[1:], _get_mlopt_sett(sa[1:], sett))
        elif sa[0] == '--knn':
            sett = _get_file_sett(sa[1:])
            return _get_knnl_sett(sa[1:], _get_mlopt_sett(sa[1:], sett))
        elif sa[0] == '--mle':
            sett = _get_file_sett(sa[1:])
            return _get_mlel_sett(sa[1:], _get_mlopt_sett(sa[1:], sett))
        elif sa[0] == '--tool':
            return _get_tool_sett(sa[1:])
        else:
            return None

"""
Command line main controller-controller function
"""
if __name__ == "__main__":
    import sys

    run_sett = _get_settings(sys.argv)
    if run_sett:
        # Switch on learner type & options
        if run_sett['pkg'] == 'dtl':
            dl = DataLoader(run_sett)
            dt_con = DTControl(run_sett)
            # AdaBoost on DTL?
            if not 'ada_iter' in run_sett:
                dt_con.run_dtl(dl)
            else:
                ada_con = AdaControl()
                ada_con.run_ada(dl, dt_con)

        elif run_sett['pkg'] == 'knn':
            dl = DataLoader(run_sett)
            knn_con = KNNControl(run_sett)
            # AdaBoost?
            if not 'ada_iter' in run_sett:
                knn_con.run_knnl(dl)
            else:
                ada_con = AdaControl()
                ada_con.run_ada(dl, knn_con)

        elif run_sett['pkg'] == 'mle':
            dl = DataLoader(run_sett)
            mle_con = MLEControl(run_sett)
            mle_con.run_mlel(dl)

        # Extra tools and stuff
        elif run_sett['pkg'] == 'tool':
            tl_con = ToolControl(run_sett)
            tl_con.run_tool()
    else:
        _print_help()

