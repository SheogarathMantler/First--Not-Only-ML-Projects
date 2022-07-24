from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline

# Data import
data = pd.read_csv('./train.csv')
# Number features
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

# Categorial features
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'
# Plotting some of the categorial features
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.pie(data["HasPhoneService"].value_counts())
ax1.legend(data["HasPhoneService"].value_counts().index)
ax2.pie(data["HasPartner"].value_counts())
ax2.legend(data["HasPartner"].value_counts().index)
plt.title('Categorial params')
plt.show()
data["HasPhoneService"].value_counts()

# 1 method : Grid search for LogRegression
categorical_data = data[cat_cols]
# One-hot encoding of the categorial features
dummy_features = pd.get_dummies(categorical_data)
X = data[feature_cols]
X = pd.concat([X[num_cols], dummy_features], axis=1)
# Replacing ' ' in the data
X['TotalSpent'] = X['TotalSpent'].replace(' ', '2287.48').astype(float)
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.8, random_state=10)
# Scaling of the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initializing of the LogRegressor
regressor = LogisticRegression()
# Grid search for the best 'C' param
CV_model = GridSearchCV(regressor, 
                        param_grid={'C':[0.05, 0.1, 0.5, 1, 2, 3, 5, 10]},
                        cv=5, 
                        scoring='f1',
                        n_jobs=-1, )
regressor = CV_model
# Training and predictions
regressor.fit(X_train_scaled, y_train)
predictions = regressor.predict_proba(X_test_scaled)
roc_auc_score(y_test, predictions[:,1]) # Result in Kaggle 0.69

# 2 method : Gradient boosting with catboost
# for catboost scaling and one-hot encoding is not needed
from catboost import CatBoostClassifier
catboost_model = CatBoostClassifier()
boost_X = data[feature_cols]
boost_X['TotalSpent'] = boost_X['TotalSpent'].replace(' ', '2287.48').astype(float)
boost_y = data[target_col]
boost_X_train, boost_X_test, boost_y_train, boost_y_test = train_test_split(boost_X, boost_y, test_size=0.8, random_state=10)
catboost_model.fit(boost_X_train, boost_y_train, cat_features=cat_cols)
boost_predictions = catboost_model.predict_proba(boost_X_test)
roc_auc_score(boost_y_test, boost_predictions[:,1]) # Result in Kaggle 0.84
