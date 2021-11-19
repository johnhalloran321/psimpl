#!/usr/bin/env python
"""
Written by John Halloran <johnhalloran321@gmail.com>

Copyright (C) 2021 John T Halloran
Licensed under the Apache License version 2.0
See http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import with_statement

import subprocess
from psimpl.psimpl_lib import *

def impute_and_write_pin(args):
    """ Imputation workflow:
        -Instantitate psm imputation object, which automatically finds missing values in a PIN file
        -Set optimization problem for regression/imputation
        -Solve optimization problem and impute values
        -Write results to an output PIN file
    """
    # Instantiate main imputation object
    pi = psm_imputer(args.pin, 
                     verb = args.verbose,
                     debug_mode = args.turn_on_debug_mode)

    # Check whether we're working with subset of features
    if args.use_subset_of_features: # Parse supplied features
        if args.features_subset:
            pi.set_feature_subset(args.features_subset)

    # Specify regression model
    print(args.impute_regressor)
    pi.set_regressor(regressor = args.impute_regressor)
    # Solve regression problem
    pi.impute()
    # Write results to output PIN
    pi.write_imputed_values(args.output_pin, args.gzip_output)

def main():
    ################ Percolator options
    parser = argparse.ArgumentParser(conflict_handler='resolve', 
                                     description="Given PIN file, impute missing (NA) feature values")
    ################ Imputation options
    imputeGroup = parser.add_argument_group('imputeGroup', 'PSIMPL options to impute missing data.')

    pinHelp = 'PIN file of PSMs for Percolator processing'
    imputeGroup.add_argument('--pin', type = str, action= 'store', default=None, help=pinHelp)

    imputeGroup.add_argument('--output-pin', type = str, action= 'store', default='psimpl.pin', help = 'PSIMPL processed PIN file.')

    imputeGroup.add_argument('--impute-regressor', type = str, action= 'store', default='LinearRegression', help = 'Regressor for imputation.')
    ################ PSIMPL options
    psimplGroup = parser.add_argument_group('psimplGroup', 'Other PSIMPL options.')

    psimplGroup.add_argument('--verbose', type = int, action= 'store', default=0, help='Specify the verbosity of the current command.')

    psimplGroup.add_argument('--turn-on-debug-mode', action='store_true', help = 'Turn off debug mode')

    psimplGroup.add_argument('--use-subset-of-features', action='store_true', help = 'Only use features specified by option --features-subset to train imputer.')

    featureSubsetHelp = "Comma delimited subset of features.  Features may either be specified name or by the numerical position they occur in the PIN header." #  For numerically specified features, a contiguous range may be specified using a colon, i.e., 1:5 specifies features 1-5 in the header file."
    imputeGroup.add_argument('--features-subset', type = str, action= 'store', default=None, help=featureSubsetHelp)

    psimplGroup.add_argument('--gzip-output', action='store_true', help = 'Compress output file using gzip.')

    _args = parser.parse_args()

    assert _args.pin != None, "Please supply Percolator PIN file to impute missing values"

    if _args.use_subset_of_features and not _args.features_subset:
        print("Use subset of features is turned on, but no subset supplied using option --features-subset.  Using all features for imputation.")
        _args.use_subset_of_features = False


    #########################
    # Run imputation worflow
    #########################
    impute_and_write_pin(_args)

if __name__ == '__main__':
    main()
