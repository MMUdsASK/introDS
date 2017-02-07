
"""
TDS2101 - INTRODUCTION TO DATA SCIENCE
ASSIGNMENT B

work.py - file to process the dataset
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", color_codes=True)


data = pd.read_csv('LoansTrainingSet.csv')
datacols = pd.Series(data.columns.values)

#-------------------------------
# DATA CLEANING PHASE ( EXECUTE ALL )
#-------------------------------

# duplicate data
dataDuplicate = data[data.duplicated(keep=False)]

# learnt that the duplicated records have the same exact attributes
del dataDuplicate

# drop the duplicates
data = data.drop_duplicates(subset="Loan ID")

# found 999999 in current loan amount
# set them to NaN so that analysis can be done on the valid values
# since the rows with the loan value all nines are fully paid,
# set the value to mean of the fully paid loans without that value
# dataL = data.loc[((data['Loan Status'] == 'Fully Paid') & (data['Current Loan Amount'] != 99999999))]
def fix4(row):
    if row['Current Loan Amount'] == 99999999:
        val = -1
    else:
        val = row['Current Loan Amount']
    return val
    
data['Current Loan Amount'] = data.apply(fix4, axis=1)

# finding invalid values in max open credit
dataScan3 = pd.to_numeric(data['Maximum Open Credit'], errors='coerce')
index = dataScan3.isnull()

# only two observations, drop
xx = data.loc[index]
data.drop(xx.index, inplace=True, axis = 0)
del dataScan3
del index
del xx

data['Maximum Open Credit'] = pd.to_numeric(data['Maximum Open Credit'])

# fixing credit score
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

# merging haveMortage to Home Mortgage
def fix3(row):
    if row['Home Ownership'] == 'HaveMortgage':
        val = 'Home Mortgage'
    else:
        val = row['Home Ownership']
    return val
    
data['Home Ownership'] = data.apply(fix3, axis=1)

# creating ordered categorical for employment
jobyears = list(data['Years in current job'].unique())
jobyearsC = ['n/a', jobyears[0], jobyears[7],jobyears[4],
             jobyears[3],
             jobyears[9], jobyears[8], jobyears[6], jobyears[5],
             jobyears[10],jobyears[2],jobyears[1]]

data['Years in current job'] = pd.Categorical(data['Years in current job'],
                                categories = jobyearsC, ordered = True)

del jobyears

## categorize loan reason
#data['Purpose'] = pd.Categorical(data['Purpose'])

# getting a workable data on debt
debt = data['Monthly Debt']
debt = debt.str.split('$',1)
data1 = [x[1] for x in debt]
data1 =[float(x.replace(',','')) for x in data1]
data['Monthly Debt Fixed'] = data1 # lost the cent, don't care
del debt
del data1

# set null values of bankrputcies to 0
data = data.dropna(subset=["Bankruptcies"])

# CHECK NULL
nan_rows = data[data.isnull().T.any().T]

# value counting
data['Annual Income'].isnull().sum()
#-------------------------------
# DATA EXPLORATION PHASE
#-------------------------------

# get column names
colnames = list(data.columns.values)

# get class distribution
data['Loan Status'].value_counts().plot(kind='bar')


# mapping employment
sns.countplot(x="Years in current job", data=data, hue = "Loan Status", palette="Greens_d");
# about the same ratio across all job lengths


# credit score, group by loan status
data.hist(column='Credit Score', by = 'Loan Status')

# boxplot credit score
sns.boxplot(y="Credit Score", x="Loan Status", data=data);
# some overlap, but there are median differences between the loan status

# mapping home ownership
sns.countplot(x="Home Ownership", data=data, hue = "Loan Status", palette="Greens_d");

# mapping purpose
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(25,15)
sns.countplot(ax=ax,x="Purpose", data=data, hue = "Loan Status", palette="Greens_d");


# HOME OWNERSHIP
home = list(data['Home Ownership'].unique())

# PURPOSE
purpose = list(data['Purpose'].unique())

# scatter matrix of all numeric attributes
sns.pairplot(data, hue="Loan Status", diag_kind="kde")
plt.savefig('out.png')

# scatter matrix for charged off
sns.pairplot(data.loc[(data['Loan Status'] != 'Fully Paid')], hue="Loan Status", diag_kind="kde")
plt.savefig('out1.png')

# moderate correlation between credit problem and tax liens
data[['Tax Liens','Number of Credit Problems']].corr()



#-------------------------------
# DATA ANALYSIS PHASE
#-------------------------------

# divide the set into short and long term
longTermLoans = data.loc[(data['Term'] == 'Long Term')]
shortTermLoans = data.loc[(data['Term'] == 'Short Term')]

# get class distribution
longTermLoans['Loan Status'].value_counts().plot(kind='bar')
# higher than normal charged off loans

shortTermLoans['Loan Status'].value_counts().plot(kind='bar')
# higher than normal charged off loans




#-------------------------------
# PREDICTION PHASE
#-------------------------------
# decision tree
# conda install scikit-learn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

dataT = pd.DataFrame()
# class attribute
dataT['Loan Status'] = data['Loan Status']
#dataT['Loan StatusB'] = preprocessing.LabelEncoder().fit_transform(dataT['Loan Status'])

# categorical attributes
ct = pd.get_dummies(data['Purpose'])
dataT = dataT.join(ct)
ct = pd.get_dummies(data['Home Ownership'])
dataT = dataT.join(ct)
ct = preprocessing.LabelEncoder().fit_transform(data['Years in current job'])
dataT['Years in current job'] = ct
dataT['Term'] = preprocessing.LabelEncoder().fit_transform(data['Term'])




# numeric attributes
dataT['Current Loan Amount'] = data['Current Loan Amount']
dataT['Credit Score'] = data['Credit Score']
dataT['Credit Score'].fillna(-1, inplace=True)
dataT['Annual Income'] = data['Annual Income']
dataT['Annual Income'].fillna(-1, inplace=True)
dataT['Monthly Debt'] = data['Monthly Debt Fixed']
dataT['Current Credit Balance'] = data['Current Credit Balance']
dataT['Maximum Open Credit'] = data['Maximum Open Credit']
dataT['Years of Credit History'] = data['Years of Credit History']
dataT['Number of Open Accounts'] = data['Number of Open Accounts']
dataT['Number of Credit Problems'] = data['Number of Credit Problems']
dataT['Current Credit Balance'] = data['Current Credit Balance']
dataT['Maximum Open Credit'] = data['Maximum Open Credit']
dataT['Bankruptcies'] = data['Bankruptcies']
# not including tax liens due to correlation with credit problem

# NULL VALUE
nan_rows = dataT[dataT.isnull().T.any().T]

# create decision tree
dt = DecisionTreeClassifier(min_samples_split=2000, random_state=99,
                            max_depth=3)

# split data into training and testing
train, test = train_test_split(dataT, train_size = 0.6)

# fit the decision tree into training data
dt.fit(train.ix[:,1:], train['Loan Status'])

# initial scoring
dt.score(test.ix[:,1:], test['Loan Status'])

# obtaining feature importance
s = pd.Series(dt.feature_importances_,index = dataT.ix[:,1:].columns)
s.sort_values(ascending=True).plot(kind='barh')
plt.title('Feature Importance')


# export decision tree
export_graphviz(dt,out_file='output1.dot',feature_names=dataT.ix[:,1:].columns,
                proportion=False,
                filled=True,
                impurity=False)
# goto webgraphviz.com to render the dot file


# showing the confusion matrix
cnf_matrix = confusion_matrix(test['Loan Status'], dt.predict(test.ix[:,1:]))
sns.heatmap(cnf_matrix, annot=True,  fmt='')
plt.xlabel('predicted value')
plt.ylabel('true value')

