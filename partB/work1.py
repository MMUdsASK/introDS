# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:50:03 2017

@author: admin
"""

import pandas as pd

data = pd.read_csv('LoansTrainingSet.csv')
datacols = pd.Series(data.columns.values)

# DATA CLEANING PHASE

# DUPLICATE DETECTED
dataDuplicate = data[data.duplicated(keep=False)]

# learnt that the duplicated records have the same exact attributes
del dataDuplicate

# drop the duplicates
data = data.drop_duplicates(subset="Loan ID")

# FOUND NON-COMPLLIANT CREDIT SCORE
# basd on original document
# all outliers have a 0 at the end

def fix1(row):
    if row['Credit Score'] > 800:
        val = row['Credit Score']/10
    else:
        val = row['Credit Score']
    return val
    
data['Credit Score'] = data.apply(fix1, axis=1)

# other and Other detected in dataset
def fix2(row):
    if row['Purpose'] == 'other':
        val = 'Other'
    else:
        val = row['Purpose']
    return val
    
data['Purpose'] = data.apply(fix2, axis=1)



# GET CLASS DISTRIBUTION
data['Loan Status'].value_counts().plot(kind='bar')


# DATA ANALYSIS PHASE
# credit score, group by loan status
data.groupby('Loan Status').hist(column='Credit Score')

# GET DISTINCT VALUES OF MANY COLUMNS:
colnames = list(data.columns.values)

# YEARS IN CURRENT JOB
jobyears = list(data['Years in current job'].unique())
# HOME OWNERSHIP
home = list(data['Home Ownership'].unique())
# PURPOSE
purpose = list(data['Purpose'].unique())



# MAXIMUM OPEN CREDIT HAS DATATYPE PROBLEM

