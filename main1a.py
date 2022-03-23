import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from gradient_descent import Objective, gradient_descent_momentum

df_train = pd.read_csv('./task1a/train.csv')
x=df_train.iloc[:,1:].to_numpy()
y=df_train.iloc[:,0].to_numpy()


lda_list=[0.1, 1, 10, 100, 200]
n_fold=10
len_fold=y.shape[0]//n_fold  #be careful if not perfect multiple

RMSE=np.zeros((len(lda_list),n_fold))
for i,lda in enumerate(lda_list):
    for k in tqdm(range(n_fold)):
        y_k=np.concatenate((y[:k*len_fold], y[(k+1)*len_fold:]))
        x_k=np.concatenate((x[:k*len_fold], x[(k+1)*len_fold:]))

        w_opt=np.linalg.inv(x_k.T@x_k + lda*np.identity(x_k.shape[1])) @ x_k.T @ y_k

        y_test=y[k*len_fold:(k+1)*len_fold]
        x_test=x[k*len_fold:(k+1)*len_fold]

        y_pred=x_test@w_opt
        RMSE[i,k]= mean_squared_error(y_test, y_pred)**0.5

print(RMSE)
RMSE=np.mean(RMSE,axis=1)
print(RMSE)
pd.DataFrame(RMSE).to_csv('./task1a/RMSE.csv', header=None, index=None)
