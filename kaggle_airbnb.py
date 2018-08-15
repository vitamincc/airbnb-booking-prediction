import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

#Loading data
df_train = pd.read_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/train_users_2.csv')
df_test = pd.read_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/test_users.csv')
labels = df_train['country_destination']
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]
df_train['age'].isnull().sum()

#%%
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)
df_all['age'].isnull().sum()

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'].isnull().sum()
df_all['age'] = np.where(df_all['age'] > 1900, 2015-df_all['age'], df_all['age'])
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

df_all_outlier = df_all[df_all['dac_year'] > df_all['tfa_year']].index.values
labels = labels.drop(df_all_outlier)
df_all = df_all[df_all['dac_year'] <= df_all['tfa_year']]

#%%One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', \
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', \
             'first_device_type', 'first_browser']

#%%
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#%%Splitting train and test
#header = list(df_all.columns.values)
piv_train = piv_train - len(df_all_outlier)
X = df_all[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)  
X_test = df_all[piv_train:]

#%%select features
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest,chi2

#X_new = SelectKBest(chi2, k=30).fit_transform(X, y)
#selector = SelectKBest(chi2, k=30)
#selector.fit(X, y)
#X_new = selector.transform(X)
#feature_name = list(X.columns[selector.get_support(indices=True)])

new_features = []

forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(X, y)
importances2 = forest.feature_importances_
model = SelectFromModel(forest, prefit=True)

#get name of selected features
feature_idx = model.get_support()
feature_name = X.columns[feature_idx].values

X_test = X_test[feature_name].values
X_new = model.transform(X)
X_new.shape

#%%
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  

#%%
xgb.fit(X_new, y)
y_pred_xgb = xgb.predict_proba(X_test)  

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_xgb[i])[::-1])[:5].tolist()

#Generate submission
sub_xgb = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_xgb.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_xgb.csv',index=False) 
#0.86509 without feature selection, 0.86435 with feature selections

#%%random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=4, random_state=0)
rf.fit(X_new, y)
importances = rf.feature_importances_
y_pred_rf = rf.predict_proba(X_test)

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_rf[i])[::-1])[:5].tolist()

sub_rf = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_rf.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_rf.csv',index=False)  
#0.85358 without feature selection, 0.85509 with feature selection

#%%neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=2, warm_start=True)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(X_new,y)
y_pred_nn = nn.predict_proba(X_test)

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_nn[i])[::-1])[:5].tolist()

sub_nn = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_nn.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_nn2.csv',index=False)  #0.85358
#0.85359 with feature selection

#%%

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
y_pred_neigh = neigh.predict_proba(X_test)

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_neigh[i])[::-1])[:5].tolist()



sub_neigh = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_neigh.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_neigh.csv',index=False)  
#0.79169 without feature selection

#%%
from sklearn import svm
sv = svm.SVC(kernel='linear', C=1)
sv.fit(X, y)
y_pred_svm = sv.predict_proba(X_test)

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_neigh[i])[::-1])[:5].tolist()

sub_sv = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_sv.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_sv.csv',index=False)  

#%% multi classes logistic regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = 'sag',
                        multi_class='multinomial',
                        C=1,
                        penalty='l2',
                        class_weight = 'balanced',
                        fit_intercept=True,
                        max_iter=80,
                        random_state=42)
lr.fit(X_new, y)
y_pred_lr = lr.predict_proba(X_test)

ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_lr[i])[::-1])[:5].tolist()

sub_lr = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_lr.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_lr.csv',index=False)  
# 0.16459 with feature selection
  

#%%model stacking
from sklearn.model_selection import StratifiedKFold

def single_model_stacking(clf):
    skf = StratifiedKFold(n_splits = 10)
    dataset_blend_train = np.zeros((X_new.shape[0],))
    dataset_blend_test_list=[]
    scores =[]
    for i, (train_index, test_index) in enumerate(skf.split(X_new, y)):
        X_train = X_new[train_index]
        y_train = y[train_index]
        X_val = X_new[test_index] 
        y_val = y[test_index]
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_val, y_val))
        dataset_blend_train[test_index] = clf.predict(X_val)
        pred = clf.predict(X_test)
        dataset_blend_test_list.append(pred)
    dataset_blend_test = np.mean(dataset_blend_test_list,axis=0)
    return dataset_blend_train,dataset_blend_test,scores
 
#%%
rf = RandomForestClassifier(max_depth=4, random_state=0)
blend_train_rf, blend_test_rf, scores_rf = single_model_stacking(rf)

nn = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=2, warm_start=True)
blend_train_nn, blend_test_nn, scores_nn = single_model_stacking(nn)

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  

blend_train_xgb, blend_test_xgb, scores_xgb = single_model_stacking(xgb)

X_blend_train = pd.concat((pd.DataFrame(blend_train_rf), pd.DataFrame(blend_train_xgb)), axis = 1) 
X_blend_test = pd.concat((pd.DataFrame(blend_test_rf), pd.DataFrame(blend_test_xgb)), axis = 1)

X_blend_train.columns = ['rf', 'xgb']
X_blend_test.columns = ['rf', 'xgb']

X_blend_train = pd.concat((pd.DataFrame(X_new), X_blend_train), axis = 1) 
X_blend_test = pd.concat((pd.DataFrame(X_test),X_blend_test), axis = 1)

#%%Classifier
xgb_blend = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  

#%%
xgb_blend.fit(X_blend_train, y)
y_pred_blend_xgb = xgb_blend.predict_proba(X_blend_test)  
#y_pred_blend_nn = nn_blend.predict_proba(X_blend_test)  


#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_blend_xgb[i])[::-1])[:5].tolist()

#Generate submission
sub_blend_xgb = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub_blend_xgb.to_csv('/Users/xucc/Documents/GMU/DAEN690/airbnb/sub_blend_xgb2.csv',index=False) 
#0.86359 with feature selection




 