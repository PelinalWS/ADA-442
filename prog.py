# %% [markdown]
# ## ADA 442 PROJECT
# 
# *   Yiğit Özarslan
# *   Batuhan İşcan
# 
# The dataset contains transactions made 

# %% [markdown]
# ## Import modules

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')
%matplotlib inline

# %% [markdown]
# ## Loading the dataset

# %%
df = pd.read_csv('./bank-additional.csv', delimiter=';')

# %%
# statistical info
df.describe()
# datatype info
df.info()

# %% [markdown]
# ## Creating a Pipeline element to specify Preprocessing 

# %%
# Define ordinal encoding for education, month, and day_of_week
ordinal_features = ["education", "month", "day_of_week"]
ordinal_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder())
])

# Define one-hot encoding for the the categorical features
categorical_features = ["job", "marital", "default", "housing", "loan", "contact", "poutcome"]
onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder())
])

# Define standard scaling for numerical features
numeric_features = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", 
                    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine ordinal and one-hot transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('onehot', onehot_transformer, categorical_features),
        ('numeric', numeric_transformer, numeric_features)
    ])

# Label encode the target variable 'y'
label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])


# %% [markdown]
# ## Split Data

# %%
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.1, test_size=0.1, random_state=42)

# %% [markdown]
# ## Models

# %%
models = [
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('classifier', XGBClassifier())
    ])
]

# %%
for model in models:
    model.fit(X_train, y_train)

# %%
param_grid_logistic = {
    'classifier__C': [0.1, 1, 10, 100]
}

param_grid_rf = {
    'classifier__n_estimators': [5, 50, 100, 200, 300]
}

param_grid_xgb = {
    'classifier__n_estimators': [50, 100, 200, 300, 500]
}
param_grids = [param_grid_logistic, param_grid_rf, param_grid_xgb]

# %%
grid_searches = []
best_scores = []
for model, param_grid in zip(models, param_grids):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    grid_searches.append(grid_search)
    best_scores.append(grid_search.best_score_)


# %%
for idx, grid_search in enumerate(grid_searches, start=1):
    print(f"Best parameters for Model {idx}: {grid_search.best_params_}")
    print(f"Best cross-validation score for Model {idx}: {grid_search.best_score_}")

# %%
for idx, grid_search in enumerate(grid_searches, start=1):
    best_model = grid_search.best_estimator_  # Get the best model from GridSearchCV
    y_pred = best_model.predict(X_test)  # Get predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate accuracy
    print(f"Model {idx} - Test Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

# %%
def extract_test_scores(cv_results):
    return pd.DataFrame({
        'test_score': cv_results['test_score']
    })

# %%
results = {}
for idx, grid_search in enumerate(grid_searches, start=1):
    best_model = grid_search.best_estimator_
    cv_results = cross_validate(best_model, X_train, y_train, cv=5, return_train_score=False, return_estimator=False, scoring='accuracy')
    results[f"Model {idx}"] = extract_test_scores(cv_results)


# %%
results_df = pd.concat(results, axis=1)
print(results_df)


# %%
best_index = best_scores.index(max(best_scores))
best_model = grid_searches[best_index].best_estimator_

# %%
import joblib

joblib.dump(best_model, 'model.pkl')

# %%
model = joblib.load('model.pkl')



