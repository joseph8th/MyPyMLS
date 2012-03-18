from dtl.controls import *

"""
Command-line and sys-level config and control
for my basic machine learning packages.

Joseph Edwards <jee8th@unm.edu>
CS429 - Spring 2012 - UNM
"""

def _print_help():
    print "Usage: mlearner [-f [*_X.dat *_Y.dat]] [-v [-3:3]] [-g [0.0:1.0]] [-i [0:]] [-l]"
    print "Data files must be stored in \'./data/\' folder, or set DATA_PATH in settings.py."

def _get_settings(args):
    sett = {}
    if len(args) < 3:
        return None
    else:
        sett['labels'] = False
        sa = args[1:]
        l = len(sa)
        i = 0
        while i < l:
            # Data file required
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
                else:
                    return None
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
            # DT gain_lambda ez-prune
            elif sa[i] in ['-g', '--gain'] and l-i > 1:
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
            # DT does file data have label header row(s)?
            elif sa[i] in ['-l', '--label']:
                sett['labels'] = True

            i += 1
        return sett

"""
Main controller-controller function
"""
if __name__ == "__main__":
    import sys
    dt_sett = _get_settings(sys.argv)
    # Only one controller to control right now (DTL)...
    if dt_sett:
        dt_con = DTControl(dt_sett)
        dt_con.run_dtl()
    else:
        _print_help()

