#My LightGBM Model. 
#Performed on lexical encoded categorical features, 
#added a subset of interaction features from the public script.
#4 fold CV ~ 1130.8
#Public LB: 1108.5
#Runtime: <1 hour on 24 core GCE instance

import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.preprocessing import LabelEncoder
from pylightgbm.models import GBMRegressor
from sklearn.decomposition import PCA
import itertools


shift = 200
COMB_FEATURE =['cat80','cat87','cat57','cat12','cat79','cat10','cat7','cat89','cat2','cat72','cat81',
               'cat11','cat1','cat13','cat9','cat3','cat16','cat90','cat23','cat36','cat73']
               

def encode(charcode):
    r = 0
    if(type(charcode) is float):
        return np.nan
    else:
        ln = len(charcode)
        for i in range(ln):
            r += (ord(charcode[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
        return r


directory = 'input/'
train = pd.read_csv(directory + 'train.csv')
test = pd.read_csv(directory + 'test.csv')

y=np.log(train.loss+shift)
tr_ids=train.id
test_ids=test.id
train.drop(['loss','id'],axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)

tr_rows=train.shape[0]
test_rows=test.shape[0]
full=pd.concat((train,test))

for c in list(train.select_dtypes(include=['object']).columns):
    if train[c].nunique() != test[c].nunique():
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

numeric_feats = train.columns.values[116:130] 

for comb in itertools.combinations(COMB_FEATURE, 2):
    full[comb[0] + "_" + comb[1]] = full[comb[0]] + full[comb[1]]
        
cats = [x for x in full.columns if 'cat' in x]
for col in cats:
    full[col] = full[col].apply(encode)
    
print(full.shape)

pca=PCA(n_components=5)
pca.fit(full.ix[:,:72])
pca_logical=pca.transform(full.ix[:,:72])
pca_logical=pd.DataFrame(pca_logical,columns=['pca_l1','pca_l2','pca_l3','pca_l4','pca_l5']).reset_index(drop=True)                                  

full=full.reset_index(drop=True)
full=pd.concat((full,pca_logical),axis=1).reset_index(drop=True)
num_1s=full.ix[:,0:51].sum(axis=1)
full['num_1s']=num_1s

train = full.iloc[:tr_rows, :].copy()
test = full.iloc[tr_rows:, :].copy()
gc.collect()

allpredictions = pd.DataFrame()
kfolds = 4  # 10 folds is better!
oof_train=np.zeros(tr_rows,)

seed = 42

#writing to file after first time execution. skip for future runs
train.to_csv('input/train_encoded3.csv',index=False)
test.to_csv('input/test_encoded3.csv',index=False)

#load prev saved encoded files. 
#train=pd.read_csv('input/train_encoded3.csv')
#test=pd.read_csv('input/test_encoded3.csv')
#tr_rows=train.shape[0]
#test_rows=test.shape[0]

allpredictions = pd.DataFrame()
kfolds = 4  # 10 folds is better!
oof_train=np.zeros(tr_rows,)

seed = 42

nbest=10000
gbmr = GBMRegressor(
    exec_path='your_LightGBM_exec_path',
    config='',
    application='regression',
    num_iterations=nbest,
    learning_rate=0.002, #0.03, 0.002
    num_leaves=200,  #180
    tree_learner='serial',
    num_threads=48,
    min_data_in_leaf=130, #125
    metric='l1',
    feature_fraction=0.27, #0.75,0.3
    feature_fraction_seed=seed,
    bagging_fraction=0.9, #0.9
    bagging_freq=5, #5
    bagging_seed=seed,
    metric_freq=50,
    verbose=0,
    #min_hessian= 5,
    max_bin=850, #850
    early_stopping_round=50 #40
)

best=[]
score=[]

kf = KFold(tr_rows, n_folds=kfolds, shuffle=True,random_state=123)
for i, (train_index, test_index) in enumerate(kf):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = train.iloc[train_index], train.iloc[test_index]
    y_train, y_val = y[train_index],y[test_index]

    gbmr.fit(X_train, y_train, test_data=[(X_val, y_val)])
    best.append(gbmr.best_round)
    oof_train[test_index]=gbmr.predict(X_val)
    scr=mean_absolute_error(np.exp(y_val)-shift,np.exp(oof_train[test_index])-shift)
    score.append(scr)
    
    allpredictions['p'+str(i)] =gbmr.predict(test)
    
    del X_train,X_val,y_train,y_val
    gc.collect()

print("Mean Abs Error:", mean_absolute_error(y_true=(np.exp(y)-shift), y_pred=(np.exp(oof_train)-shift)))

print(allpredictions.head())
print(np.mean(score))
print(np.mean(best))

submission = pd.read_csv('input/sample_submission.csv')
submission.iloc[:, 1] = np.exp(allpredictions.mean(axis=1).values)-shift
submission.to_csv('output/my_lgb1_avg.csv', index=None)

oof = pd.DataFrame(oof_train, columns=['loss'])
oof["id"] = tr_ids
oof.to_csv('output/my_lgb1_oof_tr.csv', index=False)

nbest=int(np.round(np.mean(best) * 1.25))
print(nbest)


gbmr.fit(train,y)
pred=gbmr.predict(test)
pred=(np.exp(pred)-shift)


submission = pd.read_csv(directory + 'sample_submission.csv')
submission.iloc[:, 1] = pred
submission.to_csv('output/my_lgb1.csv', index=None)
