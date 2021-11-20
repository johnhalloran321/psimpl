# Peptide-Spectrum match IMPutation Library (PSIMPL)
PSIMPL is a python library for easy imputation of missing PSM data.  Given an input PIN file, PSIMPL determines which features contain missing values.  PSIMPL then performs imputation by training a regressor on fully observed feature samples.

PSIMPL's API offers a high degree of flexibility, including a large number of state-of-the-art regression algorithms such as:
-Linear Regression
-Ridge Regression
-Lasso (i.e., L1 regularization)
-ElasticNet (i.e., L1+L2 regularization)

Future work: cross-validation (CV) for hyperparameter optimization, XGBoost/DNN/Support Vector Regression
