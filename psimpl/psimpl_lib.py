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
####   Classes
#####################################################
#####################################################
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
####   Data loading functions
#####################################################
#####################################################
def checkGzip_openfile(filename, mode = 'r'):
    if os.path.splitext(filename)[1] == '.gz':
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)

def impute_given_original_feature_matrix(feature_matrix, na_rows, na_cols,
                                         regressor = 'LinearRegression',
                                         alpha = 1.,
                                         l1_ratio = 0.5):
    """
    """
    linr = linear_model.LinearRegression(normalize = True)
    if regressor == 'Ridge':
        linr = linear_model.Ridge(alpha = alpha)
    elif regressor == 'Lasso':
        linr = linear_model.Lasso(alpha = alpha)
    elif regressor == 'ElasticNet':
        linr = linear_model.ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        
    nr, nc = feature_matrix.shape

    missing_rows = list(na_rows)
    full_rows = list([i for i in range(nr) if i not in set(na_rows)])
    nonmissing_columns = [i for i in range(nc) if i not in set(na_cols)]
    X = feature_matrix[np.ix_(full_rows, nonmissing_columns)]
    Y = feature_matrix[np.ix_(full_rows, na_cols)]
    # train regressor
    linr.fit(X,Y)
            
    # form test data
    X = feature_matrix[np.ix_(missing_rows, nonmissing_columns)]
    imputed_vals = linr.predict(X).reshape(-1)
    imputed_vals_dict = {}
    for i, ind in zip(imputed_vals, missing_rows):
        imputed_vals_dict[ind] = i
    return imputed_vals_dict
    #################### here: return imputed values and write new PIN file

def impute_and_write_pin_file(pinfile, outputpin, gzipOutput = True, regressor = 'LinearRegression'):
    """ Given Percolator PIN file and set of PSM ids, write
    output pin consisting of feature vectors only belonging to 
    given set of PSM ids
    """
    # na_tracker is a missing_value_tracker object
    na_tracker = find_missingVals(pinfile)
    # tracker contains which rows and columns have missing values
    na_rows = na_tracker.get_missing_rows()
    na_cols = na_tracker.get_missing_cols()
    na_feature_names = na_tracker.get_features_with_missing_values()

    # hashtable for rows that do not need further processing
    rows_with_na = set(na_rows)

    print("Features with missing values:")
    print(na_feature_names)
    # Load feature matrix
    X, _, psmStringInfo, row_keys = load_percolator_feature_matrix_with_nas(pinfile,
                                                                            includeBias = False, 
                                                                            na_rows = na_rows,
                                                                            na_features = na_feature_names)
    ref_key = 'spectral_contrast_angle'
    # spectral_contrast_angle
    for i, key in enumerate(row_keys):
        if key == 'spectral_contrast_angle':
            sca_col = i
            break
    
    print("Imputing missing values:")
    imputed_vals_per_na_row = impute_given_original_feature_matrix(X, 
                                                                   na_rows, na_cols)
    writeLines = set([0]) # keep track of what line numbers to write

    broken_constraints = 0
    with checkGzip_openfile(pinfile, 'r') as f:
        r = csv.DictReader(f, delimiter = '\t', skipinitialspace = True)
        headerInOrder =  r.fieldnames
        psmId_field = 'SpecId'
        if psmId_field not in headerInOrder:
            psmId_field = 'PSMId'
            if psmId_field not in headerInOrder:
                raise ValueError("No SpecId or PSMId field in PIN file %s, exitting" % (pinfile))

        nonFeatureKeys = [psmId_field, 'ScanNr', 'Label', 'Peptide'] # , 'Protein']
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
            if _impute_debug:
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
                if _impute_debug:
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
            if _impute_debug:
                print("%d imputed values, %d broken constraints" % (len(imputed_vals), broken_constraints))
                histogram(non_imputed_vals, imputed_vals, 
                          'imputed_hist.png', 
                          target_string = 'Observed values', decoy_string = 'Imputed values')
                histogram(target_imputed_vals, decoy_imputed_vals,
                          'td_imputed_hist.png', 
                          target_string = 'Target imputed values', decoy_string = 'Decoy imputed values')

    return len(writeLines)

def find_missingVals(filename, 
                     nonFeatureKeys = ['PSMId', 'Label', 'peptide', 'proteinIds'],
                     missingValueList = ['NA', 'na'],
                     load_observedVals = True):
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

def load_percolator_feature_matrix_with_nas(filename, 
                                            includeBias = True, 
                                            countUniquePeptides = False, 
                                            message = '', 
                                            na_rows = set([]),
                                            na_features = set([])):
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
    nonFeatureKeys = ['PSMId', 'Label', 'peptide', 'proteinIds']

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
        if includeBias:
            el.append(1.)
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
