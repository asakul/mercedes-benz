""" This is a simply naive XGBoost model. Python3"""

import numpy as np
import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

DATA_DIR = 'data'
TRAIN_FILE_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

raw_train_data = pd.read_csv(TRAIN_FILE_PATH)
raw_test_data = pd.read_csv(TEST_FILE_PATH)

# print(raw_train_data.head(5))

SEED = 1

train_data = raw_train_data.copy()
test_data = raw_test_data.copy()

# adopt the line below for splitting the dataset into train & val
# x_train, x_valid = train_test_split(train_data, test_size=0.2, random_state=SEED)

# a bloc of code for prediction. Need to add feature engineering above and remove 'Xn' from .drop()

xgb_params = {
    'eta': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'lambda': 8,
    'alpha': 2,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'nthread': 8
}

dtrain = xgb.DMatrix(train_data.drop(['ID','X0', 'X1','X2','X3','X4','X5','X6', 'X8', 'y'], axis=1), train_data['y'])

dtest = xgb.DMatrix(test_data.drop(['ID','X0', 'X1','X2','X3','X4','X5','X6', 'X8'], axis=1))

cv_output = xgb.cv(
    xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=50,
    verbose_eval=100, show_stdv=False
)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
plt.show()