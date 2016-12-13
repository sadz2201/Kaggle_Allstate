import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

def myScaler(DF,width,start):
    temp = DF.loss.argsort()
    ranks = np.empty(len(DF.loss), int)
    ranks[temp] = np.arange(len(DF.loss))
    ranks=np.array(ranks, dtype=float)
    
    min1=min(ranks)
    max1=max(ranks)
    perc=(ranks-min1)/(max1-min1)
    
    new_perc= perc * width + start
    DF.loss= DF.loss * new_perc
    return DF
    
A=pd.read_csv('output/my_xgb1_oof_tr.csv')
B=pd.read_csv('output/my_lgb1_oof_tr.csv')
C=pd.read_csv('output/preds_oob_keras1.csv')
D=pd.read_csv('output/preds_oob_keras2.csv') 
E=pd.read_csv('output/sub_script_oob_preds.csv') 
F=pd.read_csv('output/preds_oob_keras3.csv')

actual=pd.read_csv('input/train.csv')[['id','loss']]

A.loss=np.exp(A.loss)-200
B.loss=np.exp(B.loss)-200
E.loss=np.exp(E.loss)-200

A_t=myScaler(A.copy(),0.1,0.95)
B_t=myScaler(B.copy(),0.1,0.95)
E_t=myScaler(E.copy(),0.04,0.98) 

predictions=[]
predictions.append(A_t.loss.values) 
predictions.append(B_t.loss.values) 
predictions.append(C.loss.values) 
predictions.append(D.loss.values) 
predictions.append(E_t.loss.values) 
predictions.append(F.loss.values)

def mae_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += prediction * weight
    return mean_absolute_error(actual.loss, final_prediction)

starting_values = np.random.uniform(size=len(predictions))

cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
bounds = [(0,1)]*len(predictions)

res = minimize(mae_loss_func, 
           starting_values, 
           method = 'SLSQP', 
           bounds = bounds, 
           constraints = cons,
           options={'maxiter': 10000})

best_score = res['fun']
weights = res['x']

print('Ensemble Score: {}'.format(best_score))
print('Best Weights: {}'.format(weights))

#Apply same weights for test predictions.
#weights obtained can be slightly different everytime I run it.
#Hence I'll just use the ones obtained the 1st time I ran it.
A=pd.read_csv('output/my_xgb1.csv')
B=pd.read_csv('output/my_lgb1.csv')
C=pd.read_csv('output/sub_keras_full1.csv')
D=pd.read_csv('output/sub_keras_full2.csv') 
E=pd.read_csv('output/sub_best_script.csv') 
F=pd.read_csv('output/sub_keras_full3.csv')

submission=pd.read_csv('input/sample_submission.csv')

A_t=myScaler(A.copy(),0.1,0.95)
B_t=myScaler(B.copy(),0.1,0.95)
E_t=myScaler(E.copy(),0.04,0.98)

submission.loss=A_t.loss*0.0905 + B_t.loss*0.261 +C.loss*0.157 + D.loss*0.152 + E_t.loss*0.297 +F.loss*0.0405

submission.to_csv('output/ens_2xgb_3keras_lgbm_hack.csv',index=False)

