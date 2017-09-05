# Model that predicts the probability of default for credit applicants.
# First layer uses models such as logistic regression, random forest, and neural networks.  Finishes with a layer of xgboost for
# final predictions.  AUROC is used to evaluate model performance.

import pandas as pd
import numpy as np

# Not used unless plotting or testing, which is not done in this version
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# machine learning
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Load .csv files
train = pd.read_csv('cs-training.csv')
test = pd.read_csv('cs-test.csv')
combine = [train, test]

combine[0] = combine[0].drop(["Unnamed: 0"], axis=1)
combine[1] = combine[1].drop(["SeriousDlqin2yrs"], axis=1)

# The training set has a few people with age=0.  I found the best way to deal with them is to just drop them.
# Other approaches were to give them median age of people with same dependents or median of all ages.
combine[0] = combine[0][combine[0].age != 0]

# Replace NaNs with something else.  For monthly income, the median income of those with the same age.  Same for number
# of dependents.
guess_income = np.zeros(int(combine[0]['age'].max()+1))
guess_dependents = np.zeros(int(combine[0]['age'].max()+1))

for i in range (0, combine[0]['age'].max()+1):
    combine[0]['MonthlyIncome'].replace(1, np.NaN)
    income_df = combine[0][(combine[0]['age'] == i)]['MonthlyIncome'].dropna()
    dependents_df = combine[0][(combine[0]['age'] == i)]['NumberOfDependents'].dropna()

    income_med = income_df.median()
    dependents_med = dependents_df.median()

    if np.isnan(income_med):
        guess_income[i] = 0
    else:
        guess_income[i] = income_med

    if np.isnan(dependents_med):
        guess_dependents[i] = 0
    else:
        guess_dependents[i] = dependents_med

for ds in combine:
    for i in range (0, ds['age'].max()+1):
        ds.loc[((ds.MonthlyIncome.isnull()) | (ds.MonthlyIncome == 1)) & (ds.age == i),'MonthlyIncome'] = guess_income[i]
        ds.loc[(ds.NumberOfDependents.isnull()) & (ds.age == i), 'NumberOfDependents'] = guess_dependents[i]

    ds['MonthlyIncome'] = ds['MonthlyIncome'].astype(int)
    ds['NumberOfDependents'] = ds['NumberOfDependents'].astype(int)



for ds in combine:
    # Extract an approximation of monthly debt payments
    ds['MonthlyDebt'] = ds.DebtRatio * ds.MonthlyIncome
    ds['Income-Dep-RatioPre'] = ds.MonthlyIncome / (ds.NumberOfDependents + 1)
    ds['Income-NumLoans-RatioPre'] = ds.MonthlyIncome / (ds.NumberOfOpenCreditLinesAndLoans +
                                                      ds.NumberRealEstateLoansOrLines + 1)

combine[0]['MonthlyDebt'] = np.log1p(combine[0]['MonthlyDebt'])
combine[1]['MonthlyDebt'] = np.log1p(combine[1]['MonthlyDebt'])

combine[0]['Income-Dep-RatioPre'] = np.log1p(combine[0]['Income-Dep-RatioPre'])
combine[0]['Income-NumLoans-RatioPre'] = np.log1p(combine[0]['Income-NumLoans-RatioPre'])
combine[1]['Income-Dep-RatioPre'] = np.log1p(combine[1]['Income-Dep-RatioPre'])
combine[1]['Income-NumLoans-RatioPre'] = np.log1p(combine[1]['Income-NumLoans-RatioPre'])


# Expect some data to be normally distributed, log transform them... also keeps model inputs from getting too high
combine[0]['MonthlyIncome'] = np.log1p(combine[0]['MonthlyIncome'])
combine[0]['age'] = np.log1p(combine[0]['age'])
combine[1]['MonthlyIncome'] = np.log1p(combine[1]['MonthlyIncome'])
combine[1]['age'] = np.log1p(combine[1]['age'])


# Creating some features that improve performance.
# Relating income to dependents and current loans directly may be useful:
for ds in combine:
    ds['Income-Dep-Ratio'] = ds.MonthlyIncome / (ds.NumberOfDependents + 1)
    ds['Income-NumLoans-Ratio'] = ds.MonthlyIncome / (ds.NumberOfOpenCreditLinesAndLoans +
                                                      ds.NumberRealEstateLoansOrLines + 1)

    # Create a weighting of how late someone has been in the past. Later payment = higher weight
    ds['WeightedMissedPayments'] = 3*ds.NumberOfTimes90DaysLate + 1.5*ds['NumberOfTime60-89DaysPastDueNotWorse'] \
                                   + 0.2*ds['NumberOfTime30-59DaysPastDueNotWorse']


    # Some debt ratios are messed up and pretty high, scale them down to around what they should be
    ds.loc[(ds.DebtRatio >= 2), 'DebtRatio'] = ds.DebtRatio / 10000



# log transform other columns as it improves performance having smaller inputs.  Also not unreasonable to assume they
# are Gaussian distributed.
combine[0]['Income-Dep-Ratio'] = np.log1p(combine[0]['Income-Dep-Ratio'])
combine[0]['Income-NumLoans-Ratio'] = np.log1p(combine[0]['Income-NumLoans-Ratio'])
combine[0]['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(combine[0]['RevolvingUtilizationOfUnsecuredLines'])
combine[0]['DebtRatio'] = np.log1p(combine[0]['DebtRatio'])
combine[0]['NumberOfOpenCreditLinesAndLoans'] = np.log1p(combine[0]['NumberOfOpenCreditLinesAndLoans'])
combine[0]['NumberRealEstateLoansOrLines'] = np.log1p(combine[0]['NumberRealEstateLoansOrLines'])
combine[0]['NumberOfDependents'] = np.log1p(combine[0]['NumberOfDependents'])
combine[0]['WeightedMissedPayments'] = np.log1p(combine[0]['WeightedMissedPayments'])
combine[1]['Income-Dep-Ratio'] = np.log1p(combine[1]['Income-Dep-Ratio'])
combine[1]['Income-NumLoans-Ratio'] = np.log1p(combine[1]['Income-NumLoans-Ratio'])
combine[1]['RevolvingUtilizationOfUnsecuredLines'] = np.log1p(combine[1]['RevolvingUtilizationOfUnsecuredLines'])
combine[1]['DebtRatio'] = np.log1p(combine[1]['DebtRatio'])
combine[1]['NumberOfOpenCreditLinesAndLoans'] = np.log1p(combine[1]['NumberOfOpenCreditLinesAndLoans'])
combine[1]['NumberRealEstateLoansOrLines'] = np.log1p(combine[1]['NumberRealEstateLoansOrLines'])
combine[1]['NumberOfDependents'] = np.log1p(combine[1]['NumberOfDependents'])
combine[1]['WeightedMissedPayments'] = np.log1p(combine[1]['WeightedMissedPayments'])


# Training
x_all = combine[0].drop("SeriousDlqin2yrs", axis=1)
y_all = combine[0]["SeriousDlqin2yrs"]

# Splitting up a training and test (validation) set for individually testing models
frac_test = 0.2
x_train, x_2, y_train, y_2 = train_test_split(x_all, y_all, test_size = frac_test, random_state=23)

# Training different models and testing them individually.
#LR
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
# pred_LR = logreg.predict_proba(x_2)
# pred_LR = pred_LR[:, 1]
# print("LR:", roc_auc_score(y_2, pred_LR))

# Decision Tree
dt = DecisionTreeClassifier(random_state=21)
dt.fit(x_train, y_train)
# predictions_DT = dt.predict_proba(x_2)
# predictions_DT = predictions_DT[:, 1]
# print("Decision Tree:", roc_auc_score(y_2, predictions_DT))

# Neural Net
nn = MLPClassifier(hidden_layer_sizes=(7,4), random_state=22)
nn.fit(x_train,y_train)
nn.fit(x_train,y_train)
# predictions_NN = nn.predict_proba(x_2)
# predictions_NN = predictions_NN[:, 1]
# print("Neural Net:", roc_auc_score(y_2, predictions_NN))

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
# predictions_KNN = knn.predict_proba(x_2)
# predictions_KNN = predictions_KNN[:, 1]
# print("KNN:", roc_auc_score(y_2, predictions_KNN))

# Random Forest
rf = RandomForestClassifier(n_estimators = 200, random_state=21)
rf.fit(x_train,y_train)
# predictions_RF = rf.predict_proba(x_2)
# predictions_RF = predictions_RF[:, 1]
# print("Random Forest:", roc_auc_score(y_2, predictions_RF))

# AdaBoost
ab = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0, random_state=111)
ab.fit(x_train,y_train)
# predictions_AB = ab.predict_proba(x_2)
# predictions_AB = predictions_AB[:, 1]
# print("AdaBoost:", roc_auc_score(y_2, predictions_AB))

# Gaussian Naive-Bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
# predictions_GNB = gnb.predict_proba(x_2)
# predictions_GNB = predictions_GNB[:, 1]
# print("Gaussian Naive-Bayes:", roc_auc_score(y_2, predictions_GNB))

# RF regressor
rfreg = RandomForestRegressor(n_estimators = 200, random_state=12)
rfreg.fit(x_train,y_train)

# **Best solo performer from above is AdaBoost with AUROC of ~0.8612. A close second was the MLP, with AUROC of ~0.8574
# on the validation set.


predictions_LR_train = logreg.predict_proba(x_2)[:,1]
predictions_DT_train = dt.predict_proba(x_2)[:,1]
predictions_NN_train = nn.predict_proba(x_2)[:,1]
predictions_RFR_train = rfreg.predict(x_2)
predictions_KNN_train = knn.predict_proba(x_2)[:,1]
predictions_RF_train = rf.predict_proba(x_2)[:,1]
predictions_AB_train = ab.predict_proba(x_2)[:,1]
predictions_GNB_train = gnb.predict_proba(x_2)[:,1]

# Reshape to get the arrays to work
predictions_LR_train = predictions_LR_train.reshape(-1, 1)
predictions_DT_train = predictions_DT_train.reshape(-1, 1)
predictions_NN_train = predictions_NN_train.reshape(-1, 1)
predictions_RFR_train = predictions_RFR_train.reshape(-1, 1)
predictions_KNN_train = predictions_KNN_train.reshape(-1, 1)
predictions_RF_train = predictions_RF_train.reshape(-1, 1)
predictions_AB_train = predictions_AB_train.reshape(-1, 1)
predictions_GNB_train = predictions_GNB_train.reshape(-1, 1)

# What to train the meta model on
next_x_train = np.concatenate((predictions_LR_train,predictions_DT_train, predictions_NN_train, predictions_KNN_train,
                               predictions_RF_train, predictions_AB_train, predictions_GNB_train, predictions_RFR_train), axis=1)


meta_boost = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=100)
meta_boost.fit(next_x_train, y_2)


x_test = combine[1].drop(['Unnamed: 0'], axis=1)

# First Model predictions on the test set
predictions_LR_train = logreg.predict_proba(x_test)[:,1]
predictions_DT_train = dt.predict_proba(x_test)[:,1]
predictions_NN_train = nn.predict_proba(x_test)[:,1]
predictions_RFR_train = rfreg.predict(x_test)
predictions_KNN_train = knn.predict_proba(x_test)[:,1]
predictions_RF_train = rf.predict_proba(x_test)[:,1]
predictions_AB_train = ab.predict_proba(x_test)[:,1]
predictions_GNB_train = gnb.predict_proba(x_test)[:,1]

# Reshape to get the arrays to work
predictions_LR_train = predictions_LR_train.reshape(-1, 1)
predictions_DT_train = predictions_DT_train.reshape(-1, 1)
predictions_NN_train = predictions_NN_train.reshape(-1, 1)
predictions_RFR_train = predictions_RFR_train.reshape(-1, 1)
predictions_KNN_train = predictions_KNN_train.reshape(-1, 1)
predictions_RF_train = predictions_RF_train.reshape(-1, 1)
predictions_AB_train = predictions_AB_train.reshape(-1, 1)
predictions_GNB_train = predictions_GNB_train.reshape(-1, 1)

next_x_test = np.concatenate((predictions_LR_train,predictions_DT_train, predictions_NN_train, predictions_KNN_train,
                               predictions_RF_train, predictions_AB_train, predictions_GNB_train, predictions_RFR_train), axis=1)


# Make predictions on the test set with the final model
final_pred = meta_boost.predict_proba(next_x_test)[:,1]

sub = pd.DataFrame({'Id': combine[1]['Unnamed: 0'], 'Probability': final_pred})
sub.to_csv("Credit_Risk.csv", index=False)












