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

def main():
    ################ Percolator options
    parser = argparse.ArgumentParser(conflict_handler='resolve', 
                                     description="Given PIN file, impute missing (NA) feature values")
    ################ Imputation options
    imputeGroup = parser.add_argument_group('imputeGroup', 'PRISM options to impute missing data.')

    pinHelp = 'PIN file of PSMs for Percolator processing'
    imputeGroup.add_argument('--pin', type = str, action= 'store', default=None, help=pinHelp)

    imputeGroup.add_argument('--output-pin', type = str, action= 'store', default='prism.pin', help = 'PRISM processed PIN file.')

    imputeGroup.add_argument('--impute-regressor', type = str, action= 'store', default='LinearRegression', help = 'Regressor for imputation.')
    ################ PRISM options
    psimplGroup = parser.add_argument_group('psimplGroup', 'Ohter PSIMPL options.')
    psimplGroup.add_argument('--gen-plots', action='store_true', help = 'Generate plots for the PSM support vectors.')

    psimplGroup.add_argument('--verbose', type = int, action= 'store', default=1, help='Specify the verbosity of the current command.')

    _args = parser.parse_args()

    global _verbose
    _verbose =_args.verbose

    assert _args.pin != None, "Please supply Percolator PIN file to impute missing values"

    impute_and_write_pin_file(_args.pin, 
                              _args.output_pin, 
                              gzipOutput = False, 
                              regressor = _args.impute_regressor)

if __name__ == '__main__':
    main()
