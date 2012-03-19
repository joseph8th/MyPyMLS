PACKAGE:
        MLEARNER -- My basic machine learning project.

AUTHOR:
        Joseph Edwards <jee8th@unm.edu>
        CS429 - Spring 2012 - UNM

MODULES:
        DTL -- My basic decision tree package.

USAGE:
        Usage: python mlearner.py [-f [*_X.dat *_Y.dat] | [*.arff]] [-v [-3:3]] [-g [0.0:1.0]] [-i [0:]] [-l]
        Data files must be stored in './data/' folder, or set DATA_PATH in settings.py.

        -f or --file: filename of data file
        -v or --verb: verbosity of report output (<0 for I/O, >0 for MLearner) [def=0]
        -g or --gain: set gain 'lambda' limit for ez-pruning [def=0.0]
        -i or --iter: set # iterations of report output [def=1]
        -l or --label: process 1st data row(s) as attribute name labels [def=F]

DESCRIPTION:

'settings.py': global default settings file
'mlearner.py': executable/importable controller
'dtl/controls.py': decision tree controller
'dtl/models.py': decision tree models
'data/': default data folder (may be set in settings.py)

DTL Package:

    Reads data from either:
      (a) two '*.dat' files corresponding to the data table (X) and target labels column (Y), or
      (b) one '*.arff' (Weka) file.

    Returns decision tree. Reports gain relative to parent node for each decision node, and
    class ratios at each leaf node. More extensive node-by-node runtime reporting provided by
    (--verb) verbosity setting.

EXTRAS:
        * Supports arbitrary discrete features and class labels. 
        * Reports tree (pruned or unpruned, depending on settings) with run-time reports available by
          command line argument (--verb).

BUG REPORT:
        * My pruning function (stop at gain-lambda) sucks.
        * Used very verbose coding style (esp. for python) to get 'exploded view' of problem.
        * Stuck print fcns in everywhere.
        * Almost no documentation provided.
        * Generally ugly, but can be prettified.
