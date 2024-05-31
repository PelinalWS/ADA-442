import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv('./bank-additional.csv', delimiter=';')
dfs = df.copy()
dfl = df.copy()

ages = pd.Series(df['age'])
z_scores = np.abs(stats.zscore(ages))
filtered_ages = ages[(z_scores < 3)]
log_ages = np.log(ages + 1)
sqrt_ages = np.sqrt(ages)

fig, ax = plt.subplots(nrows= 1, ncols= 4, figsize= (20, 5))
sns.histplot(data= ages, ax= ax[0], kde= True)
sns.histplot(data= filtered_ages, ax= ax[1], kde= True)
sns.histplot(data= log_ages, ax= ax[2], kde= True)
sns.histplot(data= sqrt_ages, ax= ax[3], kde= True)

dfl['age'] = log_ages

education_categories = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
month_categories = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
day_categories = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

ordinal_features = ["education", "month", "day_of_week"]
ordinal_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder(categories=[education_categories, month_categories, day_categories]))
])

categorical_features = ["job", "marital", "default", "housing", "loan", "contact", "poutcome"]
onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder())
])

numeric_features = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", 
                    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('onehot', onehot_transformer, categorical_features),
        ('numeric', numeric_transformer, numeric_features)
    ])

label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])
dfl['y'] = label_encoder.fit_transform(dfl['y'])
dfs['y'] = label_encoder.fit_transform(dfs['y'])

xtra = []
ytra = []
xtst = []
ytst = []

X = df.drop(columns= ['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.4, test_size=0.6, random_state=42)
xtra.append(X_train)
ytra.append(y_train)
xtst.append(X_test)
ytst.append(y_test)

Xl = dfl.drop(columns= ['y'])
yl = dfl['y']

Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, train_size= 0.4, test_size=0.6, random_state=42)
xtra.append(Xl_train)
ytra.append(yl_train)
xtst.append(Xl_test)
ytst.append(yl_test)

dfs_major = dfs[dfs['y'] == 0]
dfs_minor = dfs[dfs['y'] == 1]

dfs_m_down = resample(dfs_minor, replace= True, n_samples= int(len(dfs_major)/2), random_state= 42)
dfs_m_up = resample(dfs_major, replace= True, n_samples = int(len(dfs_minor)/2), random_state= 42)

dfs_rs = pd.DataFrame(np.concatenate([dfs_m_down, dfs_m_up]), columns = df.columns)

for col in dfs.columns:
    dfs_rs[col] = dfs_rs[col].astype(dfs[col].dtype)
    
Xs = dfs_rs.drop(columns= ['y'])
ys = dfs_rs['y']

Xs_train, Xs_test, ys_train, ys_test =  train_test_split(Xs, ys, train_size= 0.4, test_size=0.6, random_state=42)
xtra.append(Xs_train)
ytra.append(ys_train)
xtst.append(Xs_test)
ytst.append(ys_test)

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
models_fs = [
    Pipeline([
        ('preprocess', preprocessor),
        ('feature_selection', SelectFromModel(LogisticRegression())),
        ('classifier', LogisticRegression())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('feature_selection', SelectFromModel(RandomForestClassifier())),
        ('classifier', RandomForestClassifier())
    ]),
    Pipeline([
        ('preprocess', preprocessor),
        ('feature_selection', SelectFromModel(XGBClassifier())),
        ('classifier', XGBClassifier())
    ])
]

models_nor = clone(models)
models_log = clone(models)
models_res = clone(models)
models_nor_fs = clone(models_fs)
models_log_fs = clone(models_fs)
models_res_fs = clone(models_fs)

for model in models_nor:
    model.fit(xtra[0], ytra[0])
for model in models_log:
    model.fit(xtra[1], ytra[1])
for model in models_res:
    model.fit(xtra[2], ytra[2])
for model in models_nor_fs:
    model.fit(xtra[0], ytra[0])
for model in models_log_fs:
    model.fit(xtra[1], ytra[1])
for model in models_res_fs:
    model.fit(xtra[2], ytra[2])

predictions_nor = []
predictions_log = []
predictions_res = []
predictions_nor_fs = []
predictions_log_fs = []
predictions_res_fs = []

for model in models_nor:
    predictions_nor.append(model.predict(xtst[0]))
for model in models_log:
    predictions_log.append(model.predict(xtst[1]))
for model in models_res:
    predictions_res.append(model.predict(xtst[2]))
for model in models_nor_fs:
    predictions_nor_fs.append(model.predict(xtst[0]))
for model in models_log_fs:
    predictions_log_fs.append(model.predict(xtst[1]))
for model in models_res_fs:
    predictions_res_fs.append(model.predict(xtst[2]))

accuracies_nor = []
accuracies_log = []
accuracies_res = []
accuracies_nor_fs = []
accuracies_log_fs = []
accuracies_res_fs = []

for i in range(len(models)):
    accuracies_nor.append(accuracy_score(ytst[0], predictions_nor[i]))
    accuracies_log.append(accuracy_score(ytst[1], predictions_log[i]))
    accuracies_res.append(accuracy_score(ytst[2], predictions_res[i]))
    accuracies_nor_fs.append(accuracy_score(ytst[0], predictions_nor_fs[i]))
    accuracies_log_fs.append(accuracy_score(ytst[1], predictions_log_fs[i]))
    accuracies_res_fs.append(accuracy_score(ytst[2], predictions_res_fs[i]))

recall_nor = []
recall_log = []
recall_res = []
recall_nor_fs = []
recall_log_fs = []
recall_res_fs = []

for i in range(len(models)):
    recall_nor.append(recall_score(ytst[0], predictions_nor[0]))
    recall_log.append(recall_score(ytst[1], predictions_log[1]))
    recall_res.append(recall_score(ytst[2], predictions_res[2]))
    recall_nor_fs.append(recall_score(ytst[0], predictions_nor_fs[0]))
    recall_log_fs.append(recall_score(ytst[1], predictions_log_fs[1]))
    recall_res_fs.append(recall_score(ytst[2], predictions_res_fs[2]))


skf = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)

param_grid_logistic = {
    'classifier__C': [0.1, 1, 10, 100],
}

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200, 300],
}

param_grid_xgb = {
    'classifier__n_estimators': [50, 100, 200, 300],
}

param_grids = [param_grid_logistic, param_grid_rf, param_grid_xgb]

models_nor_cv = clone(models)
models_log_cv = clone(models)
models_res_cv = clone(models)
models_nor_fs_cv = clone(models_fs)
models_log_fs_cv = clone(models_fs)
models_res_fs_cv = clone(models_fs)

score_nor = []
score_log = []
score_res = []
score_nor_fs = []
score_log_fs = []
score_res_fs = []

for model in models_nor_cv:
    score_nor.append(cross_validate(model, X, y, cv=skf, return_train_score=True, return_estimator=True))
for model in models_log_cv:
    score_log.append(cross_validate(model, Xl, yl, cv=skf, return_train_score=True, return_estimator=True))
for model in models_res_cv:
    score_res.append(cross_validate(model, Xs, ys, cv=skf, return_train_score=True, return_estimator=True))
for model in models_nor_fs_cv:
    score_nor_fs.append(cross_validate(model, X, y, cv=skf, return_train_score=True, return_estimator=True))
for model in models_log_fs_cv:
    score_log_fs.append(cross_validate(model, Xl, yl, cv=skf, return_train_score=True, return_estimator=True))
for model in models_res_fs_cv:
    score_res_fs.append(cross_validate(model, Xs, ys, cv=skf, return_train_score=True, return_estimator=True))

test_nor = []
test_log = []
test_res = []
test_nor_fs = []
test_log_fs = []
test_res_fs = []

for res in score_nor:
    test_nor.append(res['test_score'])
for res in score_log:
    test_log.append(res['test_score'])
for res in score_res:
    test_res.append(res['test_score'])
for res in score_nor_fs:
    test_nor_fs.append(res['test_score'])
for res in score_log_fs:
    test_log_fs.append(res['test_score'])
for res in score_res_fs:
    test_res_fs.append(res['test_score'])

grid_searches = []
best_scores = []
for i in range(len(models_res_cv)):
    grid_search = GridSearchCV(models_res_cv[i], param_grids[i], cv= skf)
    grid_search.fit(Xs_train, ys_train)
    grid_searches.append(grid_search)
    best_scores.append(grid_search.best_score_)

for idx, grid_search in enumerate(grid_searches, start=1):
    print(f"Best parameters for Model {idx}: {grid_search.best_params_}")
    print(f"Best cross-validation score for Model {idx}: {grid_search.best_score_}")

for idx, grid_search in enumerate(grid_searches, start=1):
    best_model = grid_search.best_estimator_  # Get the best model from GridSearchCV
    y_pred = best_model.predict(X_test)  # Get predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Evaluate accuracy
    print(f"Model {idx} - Test Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

def extract_test_scores(cv_results):
    return pd.DataFrame({
        'test_score': cv_results['test_score']
    })

results = {}
for idx, grid_search in enumerate(grid_searches, start=1):
    best_model = grid_search.best_estimator_
    cv_results = cross_validate(best_model, X_train, y_train, cv=5, return_train_score=False, return_estimator=False, scoring='accuracy')
    results[f"Model {idx}"] = extract_test_scores(cv_results)


results_df = pd.concat(results, axis=1)
print(results_df)

print(np.mean(results_df['Model 1']['test_score']), np.mean(results_df['Model 2']['test_score']), np.mean(results_df['Model 3']['test_score']))
print(max(np.mean(results_df['Model 1']['test_score']), np.mean(results_df['Model 2']['test_score']), np.mean(results_df['Model 3']['test_score'])))

best_index = best_scores.index(max(best_scores))
print(best_index)
best_model = grid_searches[best_index].best_estimator_

joblib.dump(best_model, 'model.pkl')


