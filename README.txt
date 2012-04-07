PACKAGE:
        MLEARNER -- My basic machine learning project.

AUTHOR:
        Joseph Edwards <jee8th@unm.edu>
        CS429 - Spring 2012 - UNM

MODULES:
        DTL -- My basic decision tree package.
        KNNL -- My basic k-nearest neighbor package (*)
        MLEL -- My basic maximum likelihood estimator package (*)
        TOOLS -- Extra tools.

USAGE:
        python mlearner.py -h
        Data files must be stored in './data/' folder, or set DATA_PATH in settings.py.

DESCRIPTION:

'settings.py': global default settings file
'mlearner.py': executable/importable controller
'dtl/controls.py': decision tree controller
'dtl/models.py': decision tree models
'data/': default data folder (may be set in settings.py)
'knnl/knnl.py': k-nn controller
'mlel/mlel.py': mle controller
'tools/adaboost.py': adaboost controller
'tools/distroview.py': plot sigmas comparison for dimensions

ALL PACKAGES:

    Reads data from either:
      (a) two '*.dat' files corresponding to the data table (X) and target labels column (Y), or
      (b) one '*.arff' (Weka) file.
      (c) one '*.data' file and one '*.names' file in Weka header format

EXTRAS:
        * DTL Supports arbitrary discrete features and class labels. 
        * DTL Reports tree (pruned or unpruned, depending on settings) with run-time reports available by
          command line argument (--verb).
        * DTL & KNNL support adaboost option
        * MLEL plots 2D data scatter option

BUG REPORT:

    DTL:
        * My pruning function (stop at gain-lambda) sucks.
        * Used very verbose coding style (esp. for python) to get 'exploded view' of problem.
        * Stuck print fcns in everywhere.
        * Almost no documentation provided.
        * Generally ugly, but can be prettified.

    KNNL:
        * See /report/ notes for hw2

    MLEL:
        * This version is incomplete. I was unable to correctly get the error rates as required in
          the assignment spec. Am turning in code and results for what I was able to complete, and will
          email a more completed version explaining lateness when complete.

CONTACT INFO:

        GitHub source repository: 'MyPyMLS' at joseph8th
