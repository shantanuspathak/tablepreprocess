# -*- coding: utf-8 -*-
"""outlierhandling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Gua7D0v3pwDDcdniwksvwQhV-C9cNXF

# Calculate Quantile Limits using IQR
"""

def calc_limits_IQR(X_train, column, quantiles = [0.25,0.75]):
    q1,q3 = X_train[column].quantile(quantiles)
    iqr = q3 - q1
    min_val = q1 - (1.5 * iqr)
    max_val = q3 + (1.5 * iqr)
    return min_val, max_val

"""# IQR based Outlier Imputation"""

def impute_outliers(X_train,Y_train):
    for col in X_train.columns: 
        # put a condition to apply this ONLY on continuous variables
        min_l, max_l = calc_limits_IQR(X_train,col)
        X_train.loc[X_train[col] > max_l, col] = max_l
        X_train.loc[X_train[col] < min_l, col] = min_l
    return X_train,Y_train

"""# IQR Based removal of outliers"""

def remove_outliers(X_train,Y_train):
    print("Initial shape= ", X_train.shape)
    
    for col in X_train.columns:
        min_l, max_l = calc_limits_IQR(X_train,col)
        idx = X_train.index[((X_train[col] <= max_l) & (X_train[col] >= min_l))]
        X_train =  X_train.loc[idx]
        Y_train = Y_train.loc[idx]
        print("After col=",col," shape is ", X_train.shape)
    return X_train,Y_train

"""# Z score based Imputation in RAW data"""

from scipy import stats
def impute_raw_outliers_zscore(X_train,Y_train=None):
    std_vals = X_train.std()
    mean_vals = X_train.mean()
    for idx, col in enumerate(X_train.columns):
        print(col)
        # print(max(stats.zscore(X_train[col])), mean_vals[idx] + 3 * std_vals[idx])
        X_train.loc[stats.zscore(X_train[col]) < -3, col] = mean_vals[idx] - 3 * std_vals[idx]
        # print(X_train.loc[stats.zscore(X_train[col]) > 3, col])
        X_train.loc[stats.zscore(X_train[col]) > 3, col] = mean_vals[idx] + 3 * std_vals[idx]
        # if "bal" in col:
        #   break
    return X_train,Y_train

"""# Z Score based imputation for Std-scaled data"""

from scipy import stats
def impute_stdscaled_outliers_zscore(X_train,Y_train):
    for col in X_train.columns:
        X_train.loc[X_train[col] < -3, col] = -3
        X_train.loc[X_train[col] > 3, col] = 3
    return X_train,Y_train

"""# Z Score based removal for RAW data"""

from scipy import stats
def remove_raw_outliers_zscore(X_train,Y_train):
    for col in X_train.columns:
        # print("Before removing col=",col," shape is ", X_train.shape)
        idx = X_train.index[abs(stats.zscore(X_train[col])) <= 3]
        X_train =  X_train.loc[idx]
        Y_train = Y_train.loc[idx]
        print("After col=",col," shape is ", X_train.shape)
    return X_train,Y_train

"""# Z Score based removal for Std-scaled Data"""

def remove_stdscaled_outliers_zscore(X_train,Y_train):
    for col in X_train.columns:
        # print("Before removing col=",col," shape is ", X_train.shape)
        idx = X_train.index[abs(X_train[col]) <= 3]
        X_train =  X_train.loc[idx]
        Y_train = Y_train.loc[idx]
        print("After col=",col," shape is ", X_train.shape)
    return X_train,Y_train

