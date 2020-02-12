#import necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings(action='ignore')

#import dataset, csv downloaded from baseballsavant

stats_df = pd.read_csv('stats.csv')

#OLS Regression Summary

features = ['exit_velocity_avg', 'launch_angle_avg', 'barrel_batted_rate', 'hard_hit_percent', 'z_swing_percent', 'oz_swing_percent', 'meatball_swing_percent', 'pull_percent', 'opposite_percent']

X = batted_ball_df[features]
y = batted_ball_df.isolated_power
X = sm.add_constant(X)
mod = sm.OLS(y, X, hasconst= True)
res = mod.fit()
print(res.summary())

#Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_train = linreg.predict(X_train)
y_pred_test = linreg.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print('Train MSE:', train_mse)
print('Test MSE:', test_mse)

#K-fold cross validation

cv_5_results = np.mean(cross_val_score(linreg, X, y, cv=5))
cv_10_results = np.mean(cross_val_score(linreg, X, y, cv=10))
cv_15_results = np.mean(cross_val_score(linreg, X, y, cv=15))

print('CV5: ', cv_5_results)
print('CV10: ', cv_10_results)
print('CV15: ', cv_15_results)

#Scaling features

ss = StandardScaler()

X_train_stand = pd.DataFrame(ss.fit_transform(X_train), index=X_train.index)
X_train_stand.columns = X_train.columns
X_train_stand = X_train_stand[(np.abs(X_train_stand) < 3).all(axis=1)]
X_train_stand.head()