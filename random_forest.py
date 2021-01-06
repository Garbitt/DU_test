# -*- coding:utf-8

# Packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# Data
dt_train = './Data/train_auto_modified.csv'
dt_test = './Data/test_auto_modified.csv'
train = pd.read_csv(dt_train)
test = pd.read_csv(dt_test)

# Initialization of the Random Forest and the GridSearch
rfc = RandomForestClassifier()
params = {'max_depth': np.linspace(1,150, 10, dtype=int),
          'n_estimators': np.linspace(1, 300, 10, dtype=int),
          'criterion': ['gini', 'entropy']}

''' We use average precision to lead the GridSearch. It is an measure of the AUC PR'''
clf = GridSearchCV(rfc, params,
                   n_jobs=8, cv=5,
                   scoring='average_precision',
                   verbose=2)

# Separate target and data
data_train = train.drop(['INDEX', 'TARGET_FLAG', 'TARGET_AMT'], axis=1)
target_train = train['TARGET_FLAG']

data_test = test.drop(['INDEX', 'TARGET_FLAG', 'TARGET_AMT'], axis=1)
target_test = test['TARGET_FLAG']

# Create our imputer to replace missing values with the median
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp = imp.fit(data_train)

data_imp_train = imp.transform(data_train)
data_imp_test = imp.transform(data_test)

# Calibration
clf.fit(data_imp_train, target_train)

# Register performance of the best estimator
results = pd.DataFrame(clf.cv_results_)
results.drop(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params'], axis=1, inplace=True)
results.to_csv('./calibration_results.csv', index=False)

# Calculation of accident probability in the test base
test['RF_Probas'] = clf.best_estimator_.predict_proba(data_imp_test)[:,1]

# Writing the results
test.loc[:, ['INDEX', 'RF_Probas']].to_csv('./final.csv', index=False)