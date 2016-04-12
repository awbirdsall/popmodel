'''popmodel module for command line creation and integration of KineticsRun
'''
from __future__ import division
from . import main
import argparse

# command line use: HITFILE PARAMETERS [-l] LOGFILE [-o] OUTPUT [-i] IMAGE [-v]
def command():
    '''Run popmodel.main.automate() using command-line arguments.
    '''
    parser = argparse.ArgumentParser(description='integrate two- or three-'
                                                 'level LIF system for given '
                                                 'HITRAN file and set of '
                                                 'parameters')
    parser.add_argument("hitfile", help="Hitran file")
    parser.add_argument("parameters", help="YAML parameter file")
    # optional parameters
    argdict = {"logfile": "log file", "csvout": "output csv",
               "image": "output png image"}
    for arg, descr in argdict.items():
        shortflag = "-" + arg[0]
        longflag = "--" + arg
        parser.add_argument(shortflag, longflag, help=descr)
    # verbosity of logging, follow http://stackoverflow.com/a/20663028
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", const=True, default=False)
    args = parser.parse_args()

    main.automate(args.hitfile, args.parameters, args.logfile, args.csvout,
                  args.image, args.verbose)
