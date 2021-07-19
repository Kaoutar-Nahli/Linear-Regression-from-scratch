
#%%
from sklearn import datasets
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
np.random.seed(40)
from sklearn.datasets import load_boston
boston_dataset = load_boston()
X, y = datasets.load_boston(return_X_y=True)


#Cleaning and feature engineering
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.head()
#%%
boston.isnull().sum()
#%%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
#%%
correlation_matrix = boston.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)

plt.figure(figsize=(20, 5))
#%%
# Model from scratch
X, y = datasets.load_boston(return_X_y=True)
def normalize_from_scratch(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    return(X)


def shuffle_data (bx, by):
    """
    This function only shuffles the array along the first axis of a multi-dimensional array.
    The order of sub-arrays is changed but their contents remains the same.
    """
    #print('bx', bx.shape, by.shape)
    w_and_b = np.c_[bx,by] # adds y to X as a column from 506,13 to 506,14
    #print('w_and_b', w_and_b.shape)
    np.random.shuffle(w_and_b) # shuffle  returns a nontype, modifies the array I passed  
    #split  previous combined arrays t X and y
    X = w_and_b[:,:-1]
    y = w_and_b[:,-1]
    #print("shuffled shapes", X.shape,y.shape)
    return X,y

def split_train_test_from_scratch(X,y,ratio):
    idx = int((1-ratio)*len(X))
    X_train = X[:idx,:]
    y_train = y[:idx]
    X_test = X[idx:,:]
    y_test = y[idx:]
    k=[X_train, X_test, y_train, y_test]
    #print(f'number of samples in training corresponding to {ratio}',idx)
    # for i in k:
    #    print('i',i.shape)

    return X_train, X_test, y_train, y_test

class DataLoader:
    """
    Class to create mini-batch dataset for a model, shuffling the data so the graph for the loss has smoother peaks
    """
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        #X,y = shuffle_data(X,y)
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            #print('shapes in Dataloader class',batch[0].shape,batch[1].shape)
            self.batches.append(batch)
            idx += batch_size
            #print(idx)
        #np.random.shuffle(self.batches)
    

    def __getitem__(self,idx):
        #print('getitem')
        return self.batches[idx]
    def __len__(self):
        return(len(self.batches))
    




    
class LinearRegression:
    def __init__(self):
        self.w= np.random.randn(X.shape[1])
        self.b = np.random.randn()
        self.list_mse_epochs = []
        self.list_mse_batches = []
        self.idx_x_to_plot = []
        self.list_mse_validation = []
   

    def fit_mini_batches(self,X, y, X_val, y_val) :
        learning_rate = 0.001
        #divide dataset in batches of 16 samples from the dataset
        dataloader = DataLoader(X,y,16)
        self.list_mse_epochs = []
        self.list_mse_batches = []
        self.list_mse_validation = []
        self.idx_x = []
        for i in range(500):
            for x_batches, y_batches in dataloader:
                #shuffle batches
                #print('shapes before shuffling and after dividing in batches', x_batches.shape, y_batches.shape)
                #x_batches_shuffled, y_batches_shuffled = shuffle_data(x_batches, y_batches)
                x_batches_shuffled, y_batches_shuffled = x_batches, y_batches
                #prediction for the the batches using non optimal parameters yet during training 
                # to calculate the partial derivative descent and the loss function 
                y_hat = self.pred(x_batches_shuffled)
                # calculate the partial derivative
                grad_w = 2 * (y_hat - y_batches_shuffled) @ x_batches_shuffled
                grad_b = 2 * np.mean(y_hat-y_batches_shuffled)
                #updating parameters 
                self.w = self.w - learning_rate * grad_w
                self.b = self.b - learning_rate * grad_b
                #calculating and appending loss function for each batch 
                self.list_mse_batches.append(self._mse(y_batches_shuffled,y_hat))
                #print(n,i, self._mse(y_hat,y_batches_shuffled))
            #calculating and appending the mean of loss function for each epoch
            self.list_mse_epochs.append(np.mean(self.list_mse_batches[-32:]))
            self.idx_x_to_plot.append(len(self.list_mse_batches))
            self.list_mse_validation.append(self.loss_validation(X_val,y_val))
            if self.early_stopping():
                break
            #print(self._mse(y_hat,y_batches) , 'epoch', i)
    def plot_loss_training_loss_validation(self):
       # plt.plot(self.idx_x_to_plot, self.list_mse_epochs, color='red')
        #plt.plot(range(len(self.list_mse_batches)), self.list_mse_batches, color ='blue')
        plt.plot(range(len(self.list_mse_validation)), self.list_mse_validation, color='green')
    
    def loss_validation(self, X_val, y_val):
        y_pred = X_val@self.w + self.b
        return self._mse(y_pred,y_val)
        
    def early_stopping(self):
        """
        early stopping after 5 iterations
        """
        l = len(self.list_mse_validation)
        if len (self.list_mse_validation)>20:
            for i in range(5):
                if self.list_mse_validation[l-i-2]> self.list_mse_validation[l-i-1]:
                    return (False)
                return(True)

    
    def fit(self,X,y):
        learning_rate = 0.001
        for i in range (200):
            y_hat = self.pred(X)
            # calculate the partial derivative
            grad_w = 2 * (y_hat - y) @ X
            grad_b = 2 * np.mean(y_hat-y)
            #updating parameters 
            self.w = self.w - learning_rate * grad_w
            self.b = self.b - learning_rate * grad_b
    
    def _mse(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def pred(self, X):
        #print ('shape parameters and X entering predict function',X.shape, self.w.shape)
        return X@self.w + self.b

X = normalize_from_scratch(X)
X_train, X_test, y_train, y_test = split_train_test_from_scratch(X,y,0.2)
model =LinearRegression()
model.fit_mini_batches(X_train,y_train,X_test, y_test)
model.plot_loss_training_loss_validation()



  


# %%


# %%
def plot_loss_val_vs_train_vs_training_set_size(X,y):
    
    """
    behaviour of the loss function while augemnting the size of the samples 

    """
    val_loss_list = []
    train_loss_list = []
    for i in range (10,len(X)):
        print(i)
        X = X[:i,:]
        y = y[:i]
        X_train, X_test, y_train, y_test = split_train_test_from_scratch(X,y,0.2)
        model =LinearRegression()
        model.fit_mini_batches(X_train,y_train)
        y_train_pred = model.pred(X_train)
        train_loss = model._mse(y_train, y_train_pred)
        y_pred = model.pred(X_test)
        val_loss = model._mse(y_test, y_pred)
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
    plt.plot(range(len(val_loss_list)), val_loss_list, color ='red', label = 'val_loss')
    plt.plot(range(len(train_loss_list)), train_loss_list, color ='blue', label = 'train_loss')
    return(train_loss_list, val_loss_list) 

plot_loss_val_vs_train_vs_training_set_size(X,y)
#plot for loss from training set should start at zero as with two points the loss should be zero as a line can pass
#easily throuth two points.
# the loss in both plots it is close and high so the model is underfitting
