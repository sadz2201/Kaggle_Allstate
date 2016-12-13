#Keras Model3
#OHE only Categorical features (not logical), add PCA & num_1s
#use Adam optimizer, batch size 400. Different Fold seed to bring some variance
#4-fold, 8 bagged CV: ~1138
#Public LB: 1114.45
#Runtime: <1 hour on AWS GPU G2.2X instance
import numpy as np
np.random.seed(111)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0
            
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
shift=200
y = np.log(train['loss'].values+shift)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

#f_cat = [f for f in tr_te.columns if 'cat' in f]
f_cat= tr_te.columns[73:117]
tr_te_logical=tr_te.ix[:,1:73]

from sklearn.decomposition import PCA
for c in list(tr_te_logical.columns):
    tr_te_logical[c] = pd.factorize(tr_te_logical[c].values, sort=True)[0]

num_1s=num_1s=tr_te_logical.ix[:,0:51].sum(axis=1)
num_1s=np.array(num_1s/51)
num_1s=num_1s.reshape((313864,1))

pca=PCA(n_components=5)
pca.fit(tr_te_logical) 
pca_logical=pca.transform(tr_te_logical)
pca_logical=pd.DataFrame(pca_logical,columns=['pca_l1','pca_l2','pca_l3','pca_l4','pca_l5']).reset_index(drop=True)                                  

tmp = csr_matrix(tr_te_logical)
sparse_data.append(tmp)
tmp = csr_matrix(pca_logical)
sparse_data.append(tmp)

for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

tmp = csr_matrix(scaler.fit_transform(num_1s))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

nfolds = 4
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 678)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(425, input_dim = xtrain.shape[1], init = 'he_normal')) #425
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4)) #0.4
        
    model.add(Dense(200, init = 'he_normal')) #225
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.3))                 #0.3
    
    model.add(Dense(40, init = 'he_normal')) #60
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.15))                 #0.1
       
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adam')
    return(model)

i = 0
nbags = 8 #8
nepochs = 25 #55
bag_seeds=[150,250,350,450,550,650,750,850]
#bag_seeds=[111]
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        np.random.seed(bag_seeds[j])
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 400, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0], #xtr.shape[0]
                                  verbose = 2)
        pred += np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-shift
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-shift
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte)-shift, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y)-shift, pred_oob))

oob=pd.DataFrame({'id':id_train,'loss': pred_oob})
oob.to_csv('output/pred_oob_keras3.csv', index=False)

nbags = 8 #3
nepochs = 25 #55
bag_seeds=[150,250,350,450,550,650,750,850]
#pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for j in range(nbags):
    print(j)
    np.random.seed(bag_seeds[j])
    model = nn_model()
    fit = model.fit_generator(generator = batch_generator(xtrain, y, 400, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtrain.shape[0], #xtr.shape[0]
                                  verbose = 2)
    
    pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-shift
    
pred_test /= nbags
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('output/sub_keras_full3.csv', index = False)

