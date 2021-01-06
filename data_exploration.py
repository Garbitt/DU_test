# -*- coding:utf-8

# Packages
import pandas as pd
import numpy as np

# Data
dt_path = './Data/{}.csv'
file = 'test_auto'

train_dt = pd.read_csv(dt_path.format(file))
print(train_dt.head())

# Basic exploration
print(train_dt.columns)

# Numeric variables
''' We notice that some amount variables aren't detected as numeric du to the '$' '''

# INCOME, HOME_VAL, BLUEBOOK, OLDCLAIM
amount_vars = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM']

def get_amount(x):
    if x != x:
        return np.nan
    else:
        return np.float(x[1:].replace(',', ''))
train_dt[amount_vars] = train_dt[amount_vars].applymap(get_amount)

dtypes = train_dt.dtypes
num_col = dtypes.loc[(dtypes == float) | (dtypes == int)].index
print(num_col)

train_dt[num_col].describe().to_csv('./Exploration/{}_numeric_summary.csv'.format(file))

train_dt.loc[train_dt.CAR_AGE < 0, 'CAR_AGE'] = np.nan

# Quick note:
'''
We fill the nan values with the median

26% of contracts had an accident
Portfolio's mean age is 44 years old, and goes from 16 to 81 years old
1 contract has a car with a negative age (-3), we replace it with a nan
The average income is ~$62k with a median of ~$54k
'''

# Other variables
other_col = [c for c in train_dt.columns if c not in num_col]
print(other_col)

all_values = {c: list(train_dt[c].unique()) for c in other_col}
all_frequencies = {c+'_frequency': [(train_dt[c]==v).mean().round(2) for v in all_values[c]] for c in other_col}

non_num = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in all_values.items() ] +
                  [ (k,pd.Series(v)) for k,v in all_frequencies.items() ]))

non_num.sort_index(axis=1).to_csv('./Exploration/{}_non_numeric_summary.csv'.format(file))

# Quick note:
'''
None of the non numeric variables is ordered. We can replace all of them by dummies

SUV (28%) and Minivan (26%) represents more than the half of the portfolio
63% of the cars are meant for a private use
54% are female
80% of the clients are in an urban area
'''

pd.concat([train_dt[num_col],
           pd.get_dummies(train_dt[other_col], drop_first=True)],
          axis=1).to_csv('./Data/{}_modified.csv'.format(file), index=False)

'''
We are facing a binary classification problem.
Since the two classes (0 and 1) are not in equal proportions,
 the area under the curve of the precision recall (AUC PR) https://en.wikipedia.org/wiki/Precision_and_recall
 may be more relevant to use than the AUC ROC.
Some variables may be correlated and contain duplicate information. A GLM on such a short study is exluded.
We will use a Random Forest algorithm that combines good efficiency
 with a good analysis of the importance of the variables via Permutation Importance Measure (PIM).
 
Note: PIM of RF can be biased by correlation between variables.
RFE algorithm can be a solution to limit the said correlations: 
 https://www.researchgate.net/publication/258083104_Correlation_and_variable_importance_in_random_forests
'''

