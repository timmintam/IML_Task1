import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# import data from csv file
df_train = pd.read_csv('./task1a/train.csv')
# transform panda data frame to numpy lists
x=df_train.iloc[:,1:].to_numpy()
y=df_train.iloc[:,0].to_numpy()

# parameters of method
lda_list=[0.1, 1, 10, 100, 200]
n_fold=10
len_fold=y.shape[0]//n_fold  # be careful if not perfect multiple

# initialize list to store root mean squared error
RMSE=np.zeros((len(lda_list),n_fold))
# loop for different values of lambda
for i,lda in enumerate(lda_list):
    # loop on k-folds (cross-validation)
    for k in range(n_fold):
        # mask for 1 fold (remove it from traing data and test results on it)
        mask = np.ones(len(y), dtype=bool)
        mask[k::n_fold] = False

        # k training set
        y_k = y[mask]
        x_k = x[mask]

        # closed form solution for ridge regression
        w_opt=np.linalg.inv(x_k.T@x_k + lda*np.identity(x_k.shape[1])) @ x_k.T @ y_k

        # k test set
        y_test=y[~mask]
        x_test=x[~mask]

        # prediction and root mean squared error
        y_pred=x_test@w_opt
        RMSE[i,k]= mean_squared_error(y_test, y_pred)**0.5

# RMSE averaged over the 10 test folds ( for each lambda)
RMSE=np.mean(RMSE,axis=1)
print(RMSE)

# save RMSE to file
pd.DataFrame(RMSE).to_csv('./task1a/RMSE.csv', header=None, index=None)
