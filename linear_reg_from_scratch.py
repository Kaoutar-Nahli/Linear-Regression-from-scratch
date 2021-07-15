

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(40)
data = datasets.load_boston()
data.keys()
data['feature_names']
X, y = datasets.load_boston(return_X_y=True)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

class Dataloader():
    def __init__(self):
        self.batches=[]
    def divide(self,X,y, batch_size):
        for i in range(len(X)//batch_size):
            if i+batch_size<len(X):
               self.batches.append([X[i:i+batch_size],y[i:i+batch_size]])
            else:
               self.batches.append([X[i:],y[i:]])  
        return self.batches




    
class LinearRegression:
    def __init__(self):
        self.w= np.random.randn(X.shape[1])
        self.b = np.random.randn()
        self.list_mse_epochs = []
        self.list_mse_batches = []

    def fit(self,X, y) :
        learning_rate = 0.001
        dataloader = Dataloader(X,y,16)
        self.list_mse_epochs = []
        self.list_mse_batches = []
        for i in range(20):
            for x_batches, y_batches in dataloader:
                y_hat = self.pred(x_batches)
                grad_w = 2 * (y_hat - y_batches) @ x_batches
                grad_b = 2 * np.mean(y_hat-y_batches)
                self.w = self.w - learning_rate * grad_w
                self.b = self.b - learning_rate * grad_b
                self.list_mse_batches.append(self._mse(y_batches,y_hat))
            self.list_mse_epochs.append(self._mse(y_batches,y_hat))
            print(self._mse(y_hat,y_batches) , 'epoch', i)
       
    #def shuffle ():
        
    def mse(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def pred(self, X):
        return X@self.w + self.b





#%%
model =LinearRegression()
model.fit(X,y)
# %%
plt.plot(range(len(model.list_mse_epochs)), model.list_mse_epochs, color='red')
#plt.plot(range(len(model.list_mse_batches)), model.list_mse_batches)

# %%
len(model.list_mse)

# %%#%%
# %%
# - from scratch bias-variance trade-off
#   - update your from scratch linear regression code to use a validation set
#   - graph both the losses together on the same plot
#   - do they look as expected?
#   - can you identify if your model is underfitting or overfitting to the data from these graphs?
#   - which of these possibilities mean your model is biased or has high variance?
#   - add early stopping to your linear regression from scratch code


# - implement a grid search
#   - update your from scratch linear regression code to include a grid search over learning rates and batch size
#   - print the best hyperparameterisations
#   - initialise a model with them and train it
#   - save it, yes, your custom model. Does it work in the same way?