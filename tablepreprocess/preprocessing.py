# -*- coding: utf-8 -*-
"""preprocessing.ipynb

By Dr Shantanu  Pathak

For easing process of learning ML
"""

import pandas as pd

"""# Unique Value Analysis"""

def unique_value_analysis(X_train,Y_train=None):
  """ 
  Function displays all unique values of every column in X & Y
  This analysis may help in understanding Categorical and Numeric features
  """
  print("Unique values in Input X_train ::")
  for col in X_train.columns:
    print("Column name:", col)
    print("Unique Values", X_train.loc[:,col].unique())
  print("Unique values in Target Y_train ::")
  if isinstance(Y_train, pd.DataFrame):
    for col in Y_train.columns:
      print("Column name:", col)
      print("Unique Values", Y_train.loc[:,col].unique())
  elif isinstance(Y_train, pd.Series):
    print("Unique Values", Y_train.unique())
  else: # Error handling may be added
    print("Unknown type", type(Y_train))

"""# Preprocessing

## Preprocessing for Training Data
"""

from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Global variables
ordinal_column_list = None
nominal_column_list = None
scaling_type = "robust"
# valid_columns= None
# LabelEncoder_dict = None
# ohe_enc = None
# NA_fill_val = None
# cat_col_list = None
# num_col_list = None
# ScalingObj = None

def train_fill_missing(X_train, NA_fill_val):
  if NA_fill_val == "median":
    NA_fill_val = X_train.median()
    
  elif NA_fill_val == "mean":
    NA_fill_val = X_train.mean()
  elif NA_fill_val == "mode":
    NA_fill_val = X_train.mode()

  X_train.fillna(NA_fill_val, inplace=True)
  return X_train, NA_fill_val

def train_scaling(X_train,num_col_list):
  if scaling_type =="robust":
    ScalingObj = RobustScaler().fit(X_train.loc[:,num_col_list])
  elif scaling_type =="minmax":
    ScalingObj = MinMaxScaler().fit(X_train.loc[:,num_col_list])
  elif scaling_type =="standard":
    ScalingObj = StandardScaler().fit(X_train.loc[:,num_col_list])
  return ScalingObj

def training_preprocessing(X_train,cat_col_list = None, num_col_list=None, 
                  NA_list = None, NA_fill_val = "median", thres = 0.9):
  ###################################
  ### Find NA values and replace with np.nan
  print(NA_list)
  if NA_list != None:
    import numpy as np
    X_train.replace(NA_list,np.nan, inplace=True)
  print("Total missing values \n",X_train.isna().sum())

  ###################################
  ### Remove columns which have missing values more than threshold
  valid_col = X_train.columns[X_train.isnull().sum() / X_train.shape[0] < thres ]
  X_train = X_train.loc[:, valid_col]

  ###################################
  ### Remove those columns which have same value in all rows
  valid_col = X_train.columns[X_train.nunique() != 1]
  X_train = X_train.loc[:, valid_col]

  # Store list of valid columns
  valid_columns = X_train.columns

  ###################################
  ### Remove repeated rows (In case of large datasets check for repeated columns)
  X_train.drop_duplicates(inplace=True)

  ###################################
  ### Handle Categorical and Numeric columns separately
  print("Shape of X_train before Handling cat columns", X_train.shape)
  LabelEncoder_dict = None
  ohe_enc = None

  if cat_col_list == None:
    cat_col_list = X_train.select_dtypes(include="object").columns
  if num_col_list == None:
    num_col_list = set(X_train.columns).difference(set(cat_col_list))
  print("cat_col_list", cat_col_list)

  if len(cat_col_list ) > 0:
    if ordinal_column_list != None:
      from collections import defaultdict
      from sklearn.preprocessing import LabelEncoder
      LabelEncoder_dict = defaultdict(LabelEncoder)
      # Encoding the variable
      X_train.loc[:,ordinal_column_list] = X_train.loc[:,ordinal_column_list].apply(lambda x: LabelEncoder_dict[x.name].fit_transform(x))
    
    if nominal_column_list != None:
      ohe_enc = OneHotEncoder(handle_unknown='ignore',drop="first")
      ohe_enc.fit(X_train.loc[:, nominal_column_list])
      ohe_temp = ohe_enc.transform(X_train.loc[:, nominal_column_list])
      ohe_temp = pd.DataFrame(ohe_temp.todense())
      X_train_remaining = X_train.drop(nominal_column_list,axis=1)
      X_train = pd.concat([X_train_remaining,ohe_temp],axis=1)

    if nominal_column_list == None:
      print("Encoding all cat columns", cat_col_list)
      # One Hot encode all the object columns
      ohe_enc = OneHotEncoder(handle_unknown='error',drop="first")
      ohe_enc.fit(X_train.loc[:,cat_col_list])
      ohe_temp = ohe_enc.transform(X_train.loc[:, cat_col_list])
      ohe_temp = pd.DataFrame(ohe_temp.todense())
      X_train_remaining = X_train.drop(cat_col_list,axis=1)
      X_train = pd.concat([X_train_remaining,ohe_temp],axis=1)
  
  ### FIll Missing values in all columns
  X_train, NA_fill_val = train_fill_missing(X_train, NA_fill_val)

  if len(num_col_list) > 0:
    
    ScalingObj = train_scaling(X_train,num_col_list)
    X_train.loc[:,num_col_list] = ScalingObj.transform(X_train.loc[:,num_col_list])
    # print("#####", type(ScalingObj))


  if X_train.select_dtypes(include="object").empty == False:
    print("Error Some columns are still Object!!")

  print("Final shape of X_train", X_train.shape)
  
  return X_train, valid_columns, LabelEncoder_dict, ohe_enc, NA_fill_val, cat_col_list,num_col_list,ScalingObj

"""## Preprocessing for Test Data"""

def test_preprocessing(X_test, valid_columns, LabelEncoder_dict, ohe_enc, 
                       NA_fill_val, cat_col_list, num_col_list,ScalingObj,NA_list = None):
  print(valid_columns)
  # Keep ONLY valid columns as used in training
  X_test = X_test.loc[:,valid_columns]

  ###################################
  ### Find NA values and replace with np.nan
  print(NA_list)
  if NA_list != None:
    import numpy as np
    X_test.replace(NA_list,np.nan, inplace=True)
  print("Total missing values in Test \n",X_test.isna().sum())


  if LabelEncoder_dict != None:
    # Using the dictionary to label future data
    X_test.loc[:,ordinal_column_list] = X_test.loc[:,ordinal_column_list].apply(lambda x: LabelEncoder_dict[x.name].transform(x))
  
  if ohe_enc != None:
    if nominal_column_list != None:
      ohe_temp = ohe_enc.transform(X_test.loc[:, nominal_column_list])
      ohe_temp = pd.DataFrame(ohe_temp.todense())
      X_test_remaining = X_test.drop(nominal_column_list,axis=1)
      X_test = pd.concat([X_test_remaining,ohe_temp],axis=1)
    else:
      # One Hot encode all the object columns
      ohe_temp = ohe_enc.transform(X_test.loc[:,cat_col_list])
      ohe_temp = pd.DataFrame(ohe_temp.todense())
      X_test_remaining = X_test.drop(cat_col_list,axis=1)
      X_test = pd.concat([X_test_remaining,ohe_temp],axis=1)

  ### FIll Missing values in all columns using values from training data
  X_test = X_test.fillna(NA_fill_val)

  if len(num_col_list) > 0:
    X_test.loc[:,num_col_list] = ScalingObj.transform(X_test.loc[:,num_col_list])

  if X_test.select_dtypes(include="object").empty == False:
    print("Error Some columns are still Object!!")

  print("Final X_test shape", X_test.shape)
  return X_test

def evaluate_code():
  import glob
  files = glob.glob("./*.csv")
  print(files)
  for file in files:
    NA_list = None
    df = pd.read_csv(file)
    if "Emp" in file:
      target_col = "Attrition"
    elif "lung" in file:
      target_col = "class of diagnosis"
    elif "house" in file:
      target_col = "price"
    elif "bank" in file:
      target_col = "y"
    elif "heart" in file:
      target_col = "V58"
      df = df.iloc[:,:58]
      NA_list = [" -9",-18,-9]
    else:
      print("unknown file")
    
    X = df.drop(target_col, axis=1)
    Y = df.loc[:,target_col]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)
    print(file, target_col)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    # unique_value_analysis(X_train,Y_train)
    X_train, valid_columns, LabelEncoder_dict, ohe_enc, NA_fill_val, cat_col_list, num_col_list,ScalingObj = training_preprocessing(X_train, NA_list= NA_list)
    
    # Store the preprocessed file
    # file = file[2:].split(".")[0] + "_new.csv"
    # X_train.to_csv(file, index = False)

    # Transform Test data
    X_test = test_preprocessing(X_test,valid_columns, LabelEncoder_dict, ohe_enc, NA_fill_val, cat_col_list, num_col_list,ScalingObj,NA_list= NA_list)

    # Always check train and test number of columns is same
    assert X_train.shape[1] == X_test.shape[1]

if __name__ == "main" :
  evaluate_code()
