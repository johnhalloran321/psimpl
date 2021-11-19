#!/usr/bin/env python
"""
Written by John Halloran <johnhalloran321@gmail.com>

Copyright (C) 2021 John T Halloran
Licensed under the Apache License version 2.0
See http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import with_statement

import gzip
import os
import csv
import argparse
import random
import re

from sklearn import linear_model

import numpy as np
import multiprocessing as mp

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import pylab
except ImportError:
    err_print('Module "matplotlib" not available.')
    exit(-1)

_impute_debug = True

#####################################################
#####################################################
####   General plotting functions
#####################################################
#####################################################

          
def histogram(targets, decoys, output, bins = 40, prob = False, 
              target_string = 'Target Scores', decoy_string = 'Decoy Scores'):
    """Histogram of the score distribution between target and decoy PSMs.

    Arguments:
        targets: Iterable of floats, each the score of a target PSM.
        decoys: Iterable of floats, each the score of a decoy PSM.
        fn: Name of the output file. The format is inferred from the
            extension: e.g., foo.png -> PNG, foo.pdf -> PDF. The image
            formats allowed are those supported by matplotlib: png,
            pdf, svg, ps, eps, tiff.
        bins: Number of bins in the histogram [default: 40].

    Effects:
        Outputs the image to the file specified in 'output'.

    """
    l = min(min(decoys), min(targets))
    h = max(max(decoys), max(targets))
    pylab.clf()
    _, _, h1 = pylab.hist(targets, bins = bins, range = (l,h), density = prob,
                          color = 'b', alpha = 0.25)
    _, _, h2 = pylab.hist(decoys, bins = bins, range = (l,h), density = prob,
                          color = 'm', alpha = 0.25)
    pylab.legend((h1[0], h2[0]), (target_string, decoy_string), loc = 'best')
    pylab.savefig('%s' % output)

def histogram_singleDist(scores, output, xax, htitle, bins = 100, prob = False, filterAroundZero = False):
    """Histogram of a score distribution.
    """
    if filterAroundZero:
        m = np.mean(scores)
        std = np.std(scores)
        scores = [s for s in scores if abs(s) > m+3*std]
    _, _, h1 = plt.hist(scores, bins = bins, range = (min(scores),max(scores)), density = prob, color = 'b')
    plt.xlabel(xax)
    plt.title(htitle)
    plt.tight_layout()
    # plt.forceAspect(ax,aspect=1)
    plt.savefig('%s' % output, bbox_inches='tight') # , dpi=100)
    plt.clf()

#####################################################
#####################################################
####   Data loading/scanning functions
#####################################################
#####################################################
def checkGzip_openfile(filename, mode = 'r'):
    if os.path.splitext(filename)[1] == '.gz':
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def find_missingVals(filename, 
                     nonFeatureKeys = ['PSMId', 'ScanNr', 'Label', 'peptide', 'proteinIds'],
                     missingValueList = ['NA', 'na'],
                     load_observedVals = True, 
                     verb = 0):
    """ Parse Percolator PIN file and find rows/features with missing values

        For n input features and m total file fields, the file format is:
        header field 1: SpecId, or other PSM id
        header field 2: Label, denoting whether the PSM is a target or decoy
        header field 3 : Input feature 1
        header field 4 : Input feature 2
        ...
        header field n + 2 : Input feature n
        header field n + 3: Peptide, the peptide string
        header field n + 4: Protein id 1

        inputs:
        filename = PIN/tab-delimited file to load features and PSM info of
        nonFeatureKeys = fields which are not going into the feature matrix (often PSM meta info)
        missingValueList = list of possible missing value strings.  These values will be imputed and filled in
    """
    f = open(filename, 'r')
    r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    headerInOrder = r.fieldnames

    # Check header fields
    psmId_field = 'SpecId'
    if psmId_field not in headerInOrder:
        psmId_field = 'PSMId'
    nonFeatureKeys[0] = psmId_field

    peptideKey = 'peptide'
    if peptideKey not in headerInOrder:
        peptideKey = 'Peptide'
        assert peptideKey in headerInOrder, 'PIN file does not contain peptide column, exitting'
    nonFeatureKeys[2] = peptideKey

    proteinKey = 'proteinIds'
    if proteinKey not in headerInOrder:
        proteinKey = 'Proteins'
        assert proteinKey in headerInOrder, 'PIN file does not contain protein column, exitting'
    nonFeatureKeys[3] = proteinKey

    assert set(nonFeatureKeys) & set(headerInOrder), "%s does not contain proper fields (%s,%s,%s,%s,) exitting" (filename, nonFeatureKeys[0],
                                                                                                                  nonFeatureKeys[1],nonFeatureKeys[2],
                                                                                                                  nonFeatureKeys[3])
    na_tracker = missing_value_tracker(missingValueList)
    missingValues = set(missingValueList)
    # missing features and PSM IDs
    na_features = set([])
    na_psm_inds = []

    constKeys = set(nonFeatureKeys) # exclude these when reserializing data
    keys = []

    if verb > 0:
        print("Header fields for PIN file:")
        print(headerInOrder)

    for h in headerInOrder: # keep order of keys intact
        if h not in constKeys and h != '':
            keys.append(h)
    featureNames = []
    for k in keys:
        featureNames.append(k)  

    for i, l in enumerate(r):
        # proteinId field may have contained tabs
        if(None in l): 
            for extraProteinId in l[None]:
                l[proteinKey] += '\t' + extraProteinId
        psmId = l[psmId_field]
        try:
            y = int(l["Label"])
        except ValueError:
            print("Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1))
            exit(-1)
        if y != 1 and y != -1:
            print("Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1))
            exit(-2)
        el = []
        for j, k in enumerate(keys):
            try:
                el.append(float(l[k]))
            except ValueError:
                if(l[k] in missingValues):
                    el.append(0.)
                    na_tracker.found_missing_value(k, i, j, psmId)
                else:
                    print(keys)
                    print(l)
                    print("Could not convert feature %s with value %s to float, exitting" % (k, l[k]))
                    exit(-3)
    f.close()
    return na_tracker

def load_percolator_feature_matrix(filename, 
                                   countUniquePeptides = False, 
                                   message = '', 
                                   na_rows = set([]),
                                   na_features = set([]),
                                   feature_subset = None):
    """ Load Percolator feature matrix generated for each crossvalidation test bin

        For n input features and m total file fields, the file format is:
        header field 1: SpecId, or other PSM id
        header field 2: Label, denoting whether the PSM is a target or decoy
        header field 3 : Input feature 1
        header field 4 : Input feature 2
        ...
        header field n + 2 : Input feature n
        header field n + 3: Peptide, the peptide string
        header field n + 4: Protein id 1
    """
    f = open(filename, 'r')
    r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
    headerInOrder = r.fieldnames
    nonFeatureKeys = ['PSMId', 'Label', 'peptide', 'proteinIds'] # , 'ScanNr']

    psmId_field = 'SpecId'
    if psmId_field not in headerInOrder:
        psmId_field = 'PSMId'
    nonFeatureKeys[0] = psmId_field
    
    peptideKey = 'peptide'
    if peptideKey not in headerInOrder:
        peptideKey = 'Peptide'
        assert peptideKey in headerInOrder, 'PIN file does not contain peptide column, exitting'
    nonFeatureKeys[2] = peptideKey

    proteinKey = ''
    if 'Protein' in headerInOrder:
        proteinKey = 'Protein'
    elif 'Proteins' in headerInOrder:
        proteinKey = 'Proteins'
    elif 'proteinIds' in headerInOrder:
        proteinKey = 'proteinIds'
    else:
        print("Protein field missing, exitting")
        exit(-1)
    nonFeatureKeys[3] = proteinKey

    assert set(nonFeatureKeys) & set(headerInOrder), "%s does not contain proper fields (%s,%s,%s,%s,) exitting" (filename, nonFeatureKeys[0],
                                                                                                                  nonFeatureKeys[1],nonFeatureKeys[2],
                                                                                                                  nonFeatureKeys[3])
    uniquePeptides = set([])

    constKeys = set(nonFeatureKeys) # exclude these when reserializing data
    keys = []
    for h in headerInOrder: # keep order of keys intact
        if h not in constKeys and h!= '':
            keys.append(h)

    if feature_subset and not feature_subset.is_empty:
        keys = feature_subset.return_overlapping_features(keys)
        print("Overlapping features:")
        print(keys)
    featureNames = []
    for k in keys:
        featureNames.append(k)

    features = []
    Y = []
    psmStringInfo = []
    for i, l in enumerate(r):
        # proteinId field may have contained tabs
        if(None in l): 
            for extraProteinId in l[None]:
                l[proteinKey] += '\t' + extraProteinId
        psmId = l[psmId_field]
        try:
            y = int(l["Label"])
        except ValueError:
            print("Could not convert label %s on line %d to int, exitting" % (l["Label"], i+1))
            exit(-1)
        if y != 1 and y != -1:
            print("Error: encountered label value %d on line %d, can only be -1 or 1, exitting" % (y, i+1))
            exit(-2)
        el = []
        for k in keys:
            if i in na_rows and k in na_features:
                el.append(0.)
            else:
                try:
                    el.append(float(l[k]))
                except ValueError:
                    print("Could not convert feature %s with value %s to float, exitting" % (k, l[k]))
                    exit(-3)
        psmInfo = PSM(l[psmId_field], l[peptideKey], l[proteinKey])
        if countUniquePeptides:
            uniquePeptides.add(l[peptideKey])
        features.append(el)
        Y.append(y)
        psmStringInfo.append(psmInfo)
    f.close()

    if countUniquePeptides:
        if message:
            print(message)
        print("Loaded %d PSMs, %d unique Peptides" % (len(psmStringInfo), len(uniquePeptides)))

    return np.array(features), np.array(Y), psmStringInfo, keys

#####################################################
#####################################################
####   Classes
#####################################################
#####################################################
class simple_feature_string_collection(object):
    """ Parse and keep track of specified features
    """
    def __init__(self):
        self.feature_inds = []
        self.feature_strings = []

    def parse_feature_subset(self, subset_string):
        """ Given comma-delimited string, parse into numerical and string values
        """
        feature_inds = self.feature_inds
        feature_names = self.feature_strings
        if not subset_string:
            return None
        for feature in subset_string.split(','):
            if feature == '':
                continue
            try:
                ind = int(feature)
                feature_inds.append(ind)
            except ValueError:
                if not feature.isspace():
                    feature_names.append(feature)

    def add_feature_names(self, list_of_features):
        """ Given listof feature names, add to feature_strings list
        """
        feature_names = self.feature_strings
        feature_name_hash = set(self.feature_strings)
        for feature in list_of_features:
            if feature not in feature_name_hash:
                feature_names.append(feature)

    @property
    def print(self):
        if self.feature_inds:
            print("Parsed numerical feature values:")
            print(self.feature_inds)
        if self.feature_strings:
            print("Parsed feature names:")
            print(self.feature_strings)

    @property
    def is_empty(self):
        if not self.feature_inds and not self.feature_strings:
            return True
        return False

    def return_overlapping_features(self, feature_names):
        """ Given list of feature names, return corresponding subset of features
        """
        subset_of_features = []
        feature_ind_hash = set(self.feature_inds)
        feature_string_hash = set(self.feature_strings)
        for i,feature in enumerate(feature_names):
            if i in feature_ind_hash or feature in feature_string_hash:
                subset_of_features.append(feature)
        return subset_of_features

class missing_value_tracker(object):
    """ Class detailing missing value info for feature matrices
    """
    def __init__(self, missingValueList = ['NA', 'na']):
        self.missingValues = set(missingValueList)
        self.features = set([])
        self.feature_mat_indices_psmIds = [] # row and column info for missing values

    def found_missing_value(self, feature_name, row, col, psmId):
        self.features.add(feature_name)
        self.feature_mat_indices_psmIds.append((row,col, psmId))

    def get_missing_cols(self):
        return list(set([j for (_,j, _) in self.feature_mat_indices_psmIds]))

    def get_missing_rows(self):
        return [i for (i,_, _) in self.feature_mat_indices_psmIds]

    def get_missing_psmIds(self):
        return [k for (_,_, k) in self.feature_mat_indices_psmIds]

    def get_features_with_missing_values(self):
        return self.features

class PSM(object):
    """ Simple class to store PSM string info
        Add scan+expMass as a PSM's hash value
    """

    def __init__(self, psmId = '', sequence = '', protein = ''):
        self.peptide = sequence
        # self.peptide = re.sub("[\[].*?[\]]", "", sequence) # strip any modifications
        # # TODO: preserve this info and add it back in later
        # self.protein = protein
        self.psmId = psmId
        self.left_flanking_aa = ''
        self.right_flanking_aa = ''

        # Check if there were multiple proteins
        l = protein.split('\t')
        if len(l) == 1:
            self.protein = l[0]
        elif len(l) > 1:
            self.protein = set(l)
        else:
            raise ValueError("No protein(s) supplied for PSM %s, exitting" % (psmId))

        # TODO: Add support for reading modifications from an input file
        if len(sequence.split('.')) > 1: # flanking information included, split string
            s  = sequence.split('.')
            # should be 3 strings after the split
            # TODO: some checking to make sure flanking amino acids are valid
            self.left_flanking_aa = s[0]
            self.right_flanking_aa = s[-1]
            self.peptide = s[1]

    def __hash__(self):
        return hash((self.psmId, self.peptide))

    def __str__(self):
        return "%s-%s" % (self.psmId, self.peptide)

class psm_imputer(object):
    """ Imputation class
    """
    def __init__(self, pinfile, regressor = None, 
                 alpha = 1.,
                 l1_ratio = 0.5,
                 verb = 0, 
                 debug_mode = False):
        self.pinfile = pinfile
        self.regressor = regressor
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.linr = None # regression function
        self.debug_mode = debug_mode
        self.verb = verb

        if debug_mode:
            self.verb = 10 # set to max

        # Find missing values in supplied PIN file
        # na_tracker is a missing_value_tracker object
        self.na_tracker = find_missingVals(pinfile, verb = self.verb)
        # grab NA info
        self.na_rows = self.na_tracker.get_missing_rows()
        self.na_cols = self.na_tracker.get_missing_cols()
        self.na_feature_names = self.na_tracker.get_features_with_missing_values()
        if self.verb:
            print("Features with missing values:")
            print(self.na_feature_names)

        ############################
        # Imputed value
        ############################
        self.imputed_vals_dict = {}

        ###############################
        # Variables for feature subsets
        ###############################
        self.feature_subset = simple_feature_string_collection() # empty by default

        ###############################
        # Extra variables for debugging
        ###############################
        self.row_keys = [] # feature names

    def set_feature_subset(self, feature_subset_string):
        self.feature_subset.parse_feature_subset(feature_subset_string)

        # Add missing value features to the list, ensuring they are not
        # pruned when reading in the feature matrix
        self.feature_subset.add_feature_names(self.na_feature_names)

        if self.verb:
            self.feature_subset.print

    def set_regressor(self, regressor = None, 
                      alpha = None,
                      l1_ratio = None):
        if regressor:
            self.regressor = regressor
        if alpha:
            self.alpha = alpha
        if l1_ratio:
            self.l1_ratio = l1_ratio

        # Check that parameters are set
        assert self.regressor, "Regression scheme not specified, exitting"
        if regressor != 'LinearRegression':
            assert self.alpha, "Regression alpha set to none, exitting"

        if self.verb:
            print("%s regression selected" % (regressor))
        if regressor == 'Ridge':
            if self.verb > 1:
                print("Regression alpha = %f" % (self.alpha))
            self.linr = linear_model.Ridge(alpha = alpha)
        elif regressor == 'Lasso':
            if self.verb > 1:
                print("Regression alpha = %f" % (self.alpha))
            self.linr = linear_model.Lasso(alpha = alpha)
        elif regressor == 'ElasticNet':
            assert self.l1_ratio, "Regression l1_ratio set to none, exitting"
            if self.verb > 1:
                print("Regression alpha = %f, l1_ration = %f" % (self.alpha, self.l1_ratio))
            self.linr = linear_model.ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        else:
            self.linr = linear_model.LinearRegression(normalize = True)

    def given_subset_update_na_cols(self, feature_keys):
        """ Find position of na features in supplied feature_keys list
        """
        new_col_inds = []
        feature_name_hash = set(self.na_feature_names)
        for i, feature in enumerate(feature_keys):
            if feature in feature_name_hash:
                new_col_inds.append(i)

        new_col_inds.sort()
        self.na_cols = new_col_inds

    def impute(self):
        """ Perform imputation in the following steps: 
            1.) Load missing value info (*should be* performed on initialization),
            2.) Load feature matrix from pinfile
            3.) Perform imputation by solving optimization problem
        """
        feature_subset = self.feature_subset
        # input pin file
        pinfile = self.pinfile

        # Optimization problem
        linr = self.linr

        # Missing value info
        na_tracker = self.na_tracker
        na_rows = self.na_rows
        na_cols = self.na_cols
        na_feature_names = self.na_feature_names

        # Load feature matrix
        feature_matrix, _, _, row_keys = load_percolator_feature_matrix(pinfile,
                                                                        na_rows = na_rows,
                                                                        na_features = na_feature_names, 
                                                                        feature_subset = feature_subset)

        if not feature_subset.is_empty: # Update na column if using subset of features
            self.given_subset_update_na_cols(row_keys)
            na_cols = self.na_cols
        
        print("Loaded row keys")
        print(row_keys)
        nr, nc = feature_matrix.shape
        if self.verb:
            print("Finished loading feature matrix from PIN file")

        missing_rows = list(na_rows) # rows containing missing values
        full_rows = list([i for i in range(nr) if i not in set(na_rows)])
        nonmissing_columns = [i for i in range(nc) if i not in set(na_cols)]

        if self.verb >= 0:
            print("Imputing missing values")
        X = feature_matrix[np.ix_(full_rows, nonmissing_columns)]
        Y = feature_matrix[np.ix_(full_rows, na_cols)]
        # train regressor
        linr.fit(X,Y)
            
        # form test data
        X = feature_matrix[np.ix_(missing_rows, nonmissing_columns)]
        imputed_vals = linr.predict(X).reshape(-1)
        for i, ind in zip(imputed_vals, missing_rows):
            self.imputed_vals_dict[ind] = i

    def write_imputed_values(self, outputpin, gzipOutput = True):
        assert self.imputed_vals_dict != {}, "Please impute values before calling write_imputed_values.  Exitting"
        
        imputed_vals_per_na_row = self.imputed_vals_dict
        impute_debug = self.debug_mode
        # input pin file
        pinfile = self.pinfile

        # Missing value info
        na_tracker = self.na_tracker
        na_rows = self.na_rows
        na_cols = self.na_cols
        na_feature_names = self.na_feature_names

        # hashtable for rows that do not need further processing
        rows_with_na = set(na_rows)

        ref_key = 'spectral_contrast_angle'
        
        broken_constraints = 0
        with checkGzip_openfile(pinfile, 'r') as f:
            r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
            headerInOrder =  r.fieldnames
            psmId_field = 'SpecId'
            if psmId_field not in headerInOrder:
                psmId_field = 'PSMId'
                if psmId_field not in headerInOrder:
                    raise ValueError("No SpecId or PSMId field in PIN file %s, exitting" % (pinfile))

            nonFeatureKeys = [psmId_field, 'ScanNr', 'Label', 'Peptide']
            p = ''
            if 'Protein' in headerInOrder:
                p = 'Protein'
            elif 'Proteins' in headerInOrder:
                p = 'Proteins'
            else:
                print("Protein field missing, exitting")
                exit(-1)
            nonFeatureKeys.append(p)

            preKeys = [psmId_field, 'Label', 'ScanNr']
            postKeys = ['Peptide', p]
            constKeys = set(nonFeatureKeys) # exclude these when reserializing dat
            keys = []
            for h in headerInOrder: # keep order of keys intact
                if h not in constKeys:
                    keys.append(h)

            if os.path.splitext(outputpin)[1] == '.gz':
                outputpin = outputpin[:-3]

            na_feat = na_tracker.get_features_with_missing_values().pop()
            with checkGzip_openfile(outputpin, 'w') as g:
                # write new pin file header
                for k in preKeys:
                    g.write("%s\t" % k)
                for k in keys[:-1]:
                    g.write("%s\t" % k)
                # Preserve the number of tabs
                g.write("%s" % keys[-1])
                for k in postKeys:
                    g.write("\t%s" % k)
                g.write("\n")

                ####################################
                ############ Imputation debugging
                ####################################
                if impute_debug:
                    non_imputed_vals = []
                    imputed_vals = []
                    target_imputed_vals = []
                    decoy_imputed_vals = []

                for i, dict_l in enumerate(r):
                    psmId = dict_l[psmId_field]
                    if i in rows_with_na:
                        dict_l[na_feat] = imputed_vals_per_na_row[i]
                    ####################################
                    ############ Imputation debugging
                    ####################################
                    if impute_debug:
                        if i in rows_with_na:
                            imputed_vals.append(dict_l[na_feat])
                            rk = float(dict_l[ref_key])
                            if (rk != 0 and dict_l[na_feat] != 0) and dict_l[na_feat] < rk:
                                broken_constraints += 1
                                print("imputed val = %f, ref val = %f" % (dict_l[na_feat], float(dict_l[ref_key])))

                            # target/decoy distributions
                            y = int(dict_l["Label"])
                            if y == 1:
                                target_imputed_vals.append(dict_l[na_feat])
                            elif y == -1:
                                decoy_imputed_vals.append(dict_l[na_feat])
                            else:
                                print("Countered improper label on line %d" % (i))
                                exit(-1)

                        else:
                            non_imputed_vals.append(float(dict_l[na_feat]))
                        
                    for k in preKeys:
                        g.write("%s\t" % dict_l[k])
                    for k in keys[:-1]:
                        g.write("%s\t" % dict_l[k])
                    # Preserve the number of tabs
                    g.write("%s" % dict_l[keys[-1]])
                    for k in postKeys:
                        g.write("\t%s" % dict_l[k])
                    g.write("\n")

                ####################################
                ############ Imputation debugging
                ####################################
                if impute_debug:
                    print("%d imputed values, %d broken constraints" % (len(imputed_vals), broken_constraints))
                    histogram(non_imputed_vals, imputed_vals, 
                              'imputed_hist.png', 
                              target_string = 'Observed values', decoy_string = 'Imputed values')
                    histogram(target_imputed_vals, decoy_imputed_vals,
                              'td_imputed_hist.png', 
                              target_string = 'Target imputed values', decoy_string = 'Decoy imputed values')

            
            if self.verb:
                print("Wrote imputed values to output file %s" % (outputpin))
