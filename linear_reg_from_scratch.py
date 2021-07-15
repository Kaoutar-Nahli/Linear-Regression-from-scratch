#%%
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
        dataloader = Dataloader()
        dataloader = dataloader.divide(X,y,16)
        self.list_mse_epochs = []
        self.list_mse_batches = []
        n=0
        for i in range(20):
            for x_batches, y_batches in dataloader:
                y_hat = self.pred(x_batches)
                grad_w = 2 * (y_hat - y_batches) @ x_batches
                grad_b = 2 * np.mean(y_hat-y_batches)
                self.w = self.w - learning_rate * grad_w
                self.b = self.b - learning_rate * grad_b
                self.list_mse_batches.append(self._mse(y_batches,y_hat))
                n=n+1
                print(n, len(x_batches))
            self.list_mse_epochs.append(self._mse(y_batches,y_hat))
            print(self._mse(y_hat,y_batches) , 'epoch', i)
       
    #def shuffle ():
        
    def _mse(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def pred(self, X):
        return X@self.w + self.b





model =LinearRegression()
model.fit(X,y)

#plt.plot(range(len(model.list_mse_epochs)), model.list_mse_epochs, color='red')
plt.plot(range(len(model.list_mse_batches)), model.list_mse_batches)

# %%
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

class DataLoader:
    """
    Class to create mini-batch dataset for a model
    """
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            self.batches.append(batch)
            idx += batch_size
            #print(idx)

    def __getitem__(self,idx):
        #print('getitem')
        return self.batches[idx]




    
class LinearRegression:
    def __init__(self):
        self.w= np.random.randn(X.shape[1])
        self.b = np.random.randn()
        self.list_mse_epochs = []
        self.list_mse_batches = []

    def fit(self,X, y) :
        learning_rate = 0.001
        #divide dataset in batches of 16 samples from the dataset
        dataloader = DataLoader(X,y,16)
        self.list_mse_epochs = []
        self.list_mse_batches = []
        n = 0
        for i in range(20):
            for x_batches, y_batches in dataloader:
                #prediction for the the batches using non optimal parameters during training 
                # to calculate the partial derivative descent and the loss function 
                y_hat = self.pred(x_batches)
                # calculate the partial derivative
                grad_w = 2 * (y_hat - y_batches) @ x_batches
                grad_b = 2 * np.mean(y_hat-y_batches)
                #updating parameters 
                self.w = self.w - learning_rate * grad_w
                self.b = self.b - learning_rate * grad_b
                #calculating and appending loss function for each batch 
                n+=1
                self.list_mse_batches.append(self._mse(y_batches,y_hat))
                print(n)
            #calculating and appending loss function for each epoch
            self.list_mse_epochs.append(self._mse(y_batches,y_hat))
            print(self._mse(y_hat,y_batches) , 'epoch', i)
       
    #def shuffle ():
        
    def _mse(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def pred(self, X):
        return X@self.w + self.b



model =LinearRegression()
model.fit(X,y)

#plt.plot(range(len(model.list_mse_epochs)), model.list_mse_epochs, color='red')
plt.plot(range(len(model.list_mse_batches)), model.list_mse_batches)
