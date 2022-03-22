import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from gradient_descent import Objective, gradient_descent_momentum

df_train = pd.read_csv('./task1a/train.csv')
x=df_train.iloc[:,1:].to_numpy()
y=df_train.iloc[:,0].to_numpy()


class ToyObjective(Objective):
    def __init__(self, x, y, lda):
        self.x = x
        self.y = y
        self.lda = lda

    def __call__(self, w):
        xw=self.x@w
        #return np.transpose(xw)@xw-2*np.transpose(xw)@y+np.transpose(y)@y + self.lda*np.transpose(w)@w
        return np.sum(np.square((self.y - self.x@w))) + self.lda*np.sum(np.square(w))


    def grad(self, w):
        return  2*np.sum(-np.transpose(np.transpose(self.x)*(self.y - self.x@w)), axis=0) + 2*lda*w

lda_list=[0.1, 1, 10, 100, 200]
n_fold=10
len_fold=y.shape[0]//n_fold  #be careful if not perfect multiple

RMSE=np.zeros((len(lda_list),n_fold))
w_opt_list=np.zeros((len(lda_list),n_fold,x.shape[1]))
for i,lda in enumerate(lda_list):
    for k in tqdm(range(n_fold)):
        y_k=np.concatenate((y[:k*len_fold], y[(k+1)*len_fold:]))
        x_k=np.concatenate((x[:k*len_fold], x[(k+1)*len_fold:]))
        obj = ToyObjective(x_k,y_k,lda)
        egv,evec=np.linalg.eig(np.transpose(x_k)@x_k)
        eta=1/max(egv)
        learning_rate = 0.9*eta
        tol = 1e-10
        n_steps = 100000
        w_init = np.ones(13)
        normalize = False

        results = gradient_descent_momentum(obj, w_init, learning_rate=learning_rate, tol=tol, n_steps=n_steps, normalize=normalize)
        w_opt=results[0]
        w_opt_list[i,k,:]=w_opt

        y_test=y[k*len_fold:(k+1)*len_fold]
        x_test=x[k*len_fold:(k+1)*len_fold]

        y_pred=x_test@w_opt
        RMSE[i,k]= mean_squared_error(y_test, y_pred)**0.5

    pd.DataFrame(w_opt_list[i]).to_csv(f'./task1a/w_opt_lda={lda}.csv', header=None, index=None)

print(RMSE)
RMSE=np.mean(RMSE,axis=1)
print(RMSE)
pd.DataFrame(RMSE).to_csv('./task1a/RMSE.csv', header=None, index=None)
