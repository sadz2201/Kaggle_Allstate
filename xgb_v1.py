#My XGBoost Model. 
#Performed on One-hot encoded categorical features.
#4 fold CV ~ 1132
#Public LB: 1109.3
#Runtime: <2 hours on 24 core GCE instance

import random
import pandas as pd
import numpy as np 
from sklearn.cross_validation import train_test_split, KFold
import xgboost as xgb
import operator
import gc
import scipy
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

train=pd.read_csv('input/train.csv')
test=pd.read_csv('input/test.csv')

shift=200
y=np.log(train.loss+shift)
tr_ids=train.id
test_ids=test.id
train.drop(['loss','id'],axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

tr_rows=train.shape[0]
test_rows=test.shape[0]
full=pd.concat((train,test))

for c in list(train.select_dtypes(include=['object']).columns):
    #if train[c].nunique() != test[c].nunique():
    set_train = set(train[c].unique())
    set_test = set(test[c].unique())
    remove_train = set_train - set_test
    remove_test = set_test - set_train

    remove = remove_train.union(remove_test)
    def filter_cat(x):
        if x in remove:
            return np.nan
        return x
    full[c] = full[c].apply(lambda x: filter_cat(x), 1)
        


cat_cols=full.columns.values[72:116]
for c in cat_cols:
    x=full[c].value_counts()
    x1=x[x<=5].index.values
    p2=full[full[c].isin(x1)].index.values
    full=full.set_value(p2,c,"rare")

for c in list(train.select_dtypes(include=['object']).columns):
    full[c] = pd.factorize(full[c].values, sort=True)[0]
    

pca=PCA(n_components=5)
pca.fit(full.ix[:,:72]) #72
pca_logical=pca.transform(full.ix[:,:72])
pca_logical=pd.DataFrame(pca_logical,columns=['pca_l1','pca_l2','pca_l3','pca_l4','pca_l5']).reset_index(drop=True)                                  

full=full.reset_index(drop=True)
full=pd.concat((full,pca_logical),axis=1).reset_index(drop=True)
num_1s=full.ix[:,0:51].sum(axis=1)
full['num_1s']=num_1s

full=pd.get_dummies(full,columns=cat_cols,sparse=True)
train=full[:tr_rows]
test=full[tr_rows:]

train=scipy.sparse.csc_matrix(train)
test=scipy.sparse.csc_matrix(test)

#dtrain = xgb.DMatrix(train, label=y)
dtest = xgb.DMatrix(test)
gc.collect()


xgb_params = {
    'seed': 619, #1347
    'colsample_bytree': 0.3,#0.3
    'silent': 1,
    'subsample': 0.8,
    'learning_rate': 0.003,#0.005
    'objective': 'reg:linear',
    'max_depth': 10, #10
    'min_child_weight': 25, #25
    'gamma': 1.25,
    'eval_metric' : 'mae',
    'alpha': 0.02
    
}


nrounds = 16000  
allpredictions = pd.DataFrame()
kfolds = 4  
oof_train=np.zeros(tr_rows,)
best=[]
score=[]


kf = KFold(tr_rows, n_folds=kfolds, shuffle=True,random_state=123)
for i, (train_index, test_index) in enumerate(kf):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = train[train_index], train[test_index]
    y_train, y_val = y[train_index],y[test_index]

    dtrain = xgb.DMatrix(X_train,y_train)
    dvalid = xgb.DMatrix(X_val,y_val)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbdt = xgb.train(xgb_params, dtrain, nrounds, watchlist,
                         verbose_eval=50,
                         early_stopping_rounds=50)  
    bst=gbdt.best_ntree_limit
    oof_train[test_index]=gbdt.predict(dvalid, ntree_limit=bst)
    scr=mean_absolute_error(np.exp(y_val)-shift,np.exp(oof_train[test_index])-shift)
    
    best.append(bst)    
    score.append(scr)
    
    
    del dtrain
    del dvalid
    del gbdt
    gc.collect()

print("Mean Abs Error:", mean_absolute_error(y_true=(np.exp(y)-shift), y_pred=(np.exp(oof_train)-shift)))

print(np.mean(score))
print(np.mean(best))

oof = pd.DataFrame(oof_train, columns=['loss'])
oof["id"] = tr_ids
oof.to_csv('output/my_xgb1_oof_tr.csv', index=False)

best_nrounds=int(round(np.mean(best) * 1.25))
dtrain=xgb.DMatrix(train,y)

watchlist = [(dtrain, 'train')]
gbdt = xgb.train(xgb_params, dtrain, best_nrounds,watchlist,verbose_eval=50,early_stopping_rounds=50)
log_pred=gbdt.predict(dtest)
pred=np.exp(log_pred)-shift

sample=pd.read_csv('input/sample_submission.csv')
sample.loss=pred
sample.to_csv('output/my_xgb1.csv',index=False)
