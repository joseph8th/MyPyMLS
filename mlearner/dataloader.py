from settings import *
from string import *

class DataLoader:

    def __init__(self, sett):
        self.sett = sett
        # Default settings for DT control in settings.py
        if 'verb' not in sett:
            sett['verb'] = DEF_DT_VERB

    # Split strings by whitespace or comma
    def _split_string(self, data):
        if data[0].find(',') != -1:
            data = map(lambda c: c.split(','), data)
        else:
            data = map(lambda c: c.split(), data)
        if self.sett['verb'] < -2: 
            print "\nSplit data:\n==================\n", data
        return data

    # Cleanup and format data table for DTree
    def _format_records(self, data):
        dl = [] 
        label_row = map(lambda c: c.strip(), data[0])
        for record in data[1:]:
            record = map(lambda c: c.strip(), record)
            dl.append(dict([ (label_row[col], record[col]) for col in range(len(record)) ]))
        if self.sett['verb'] < -2:
            print "\nFormatted records:\n=================\n", dl
        return dl

    # Load file from disk
    def _load_data(self, fname):
        f = open(DATA_PATH + fname)
        D = f.readlines()
        f.close()
        if self.sett['verb'] < -2:
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

    def _get_arff_meta(self, data):
        meta_l = self._split_string([l for l in data if l.startswith('@')])
        target = [r[1] for r in meta_l if r[0] == '@relation'][0]
        label_row = [a[1] for a in meta_l if a[0] == '@attribute']
        return (target, label_row)

    def _unpack_XY(self, data, target, label_row):
        data.insert(0, label_row)
        data = self._format_records(data)
        X = data[:]
        Y = {target : [X[ix].pop(target) for ix in range(len(X))]}
        return (X, Y)

    def _load_arff(self, dfile):
        print "\nLoading data from {}.".format(dfile)
        D = self._load_data(dfile)
        target, label_row = self._get_arff_meta(D)
        strip_D = map(strip, D)
        dix = [strip_D.index(x) for x in strip_D if x.startswith('@data')][0]
        strip_D = [strip_D[i] for i in range(dix + 1, len(strip_D)) if not strip_D[i].startswith('%')]
        strip_D = self._split_string(strip_D)
        return self._unpack_XY(strip_D, target, label_row)

    # Since uci .names files not standard, must be in .arff header format
    def _load_uci(self, (dfile, nfile)):
        print "\nLoading data from {} and names from {} (arff meta format).".format(dfile, nfile)
        meta_l = self._load_data(nfile)
        target, label_row = self._get_arff_meta(meta_l)
        D = self._split_string(self._load_data(dfile))
        return self._unpack_XY(D, target, label_row)


    """
    Main method to return X, Y, & W to ML controller
    """
    def get_XYW(self):

        if self.sett['ftype'] == 'dat':
            X,Y = self._load_dat(self.sett['ftuple'], self.sett['labels'])
        elif self.sett['ftype'] == 'arff':
            X,Y = self._load_arff(self.sett['ftuple'])
        elif self.sett['ftype'] == 'uci':
            X,Y = self._load_uci(self.sett['ftuple'])
        else:
            return None

        # custom input weights or default? ...
        if 'wfile' not in self.sett:
            W = [1.0 for _ in range(len(Y.values()[0]))]
        else:
            W = self._split_string(self._load_data(self.sett['wfile']))
            W = [float(r[0]) for r in W] 

        return (X, Y, W)
