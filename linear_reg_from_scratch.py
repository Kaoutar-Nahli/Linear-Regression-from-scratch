
#%%
from sklearn import datasets
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
np.random.seed(40)
data = datasets.load_boston()
data.keys()
data['feature_names']
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
    for i in k:
       print('i',i.shape)

    return X_train, X_test, y_train, y_test

class DataLoader:
    """
    Class to create mini-batch dataset for a model, shuffling the data so the graph for the loss has smoother peaks
    """
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        X,y = shuffle_data(X,y)
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            #print('shapes in Dataloader class',batch[0].shape,batch[1].shape)
            self.batches.append(batch)
            idx += batch_size
            #print(idx)
        np.random.shuffle(self.batches)
    

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

   

    def fit(self,X, y) :
        learning_rate = 0.001
        #divide dataset in batches of 16 samples from the dataset
        dataloader = DataLoader(X,y,16)
        self.list_mse_epochs = []
        self.list_mse_batches = []
        n = 0
        for i in range(20):
            for x_batches, y_batches in dataloader:
                #shuffle batches
                #print('shapes before shuffling and after dividing in batches', x_batches.shape, y_batches.shape)
                x_batches_shuffled, y_batches_shuffled = shuffle_data(x_batches, y_batches)

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
                n+=1
                self.list_mse_batches.append(self._mse(y_batches_shuffled,y_hat))
                print(n,i, self._mse(y_hat,y_batches_shuffled))
            #calculating and appending loss function for each epoch
            self.list_mse_epochs.append(self._mse(y_batches,y_hat))
            #print(self._mse(y_hat,y_batches) , 'epoch', i)
       
    

        
    def _mse(self, y, y_pred):
        return np.mean((y-y_pred)**2)

    def pred(self, X):
        #print ('shape parameters and X entering predict function',X.shape, self.w.shape)
        return X@self.w + self.b

X = normalize_from_scratch(X)
X_train, X_test, y_train, y_test = split_train_test_from_scratch(X,y,0.2)

model =LinearRegression()
model.fit(X_train,y_train)

plt.plot(range(len(model.list_mse_epochs)), model.list_mse_epochs, color='red')
#plt.plot(range(len(model.list_mse_batches)), model.list_mse_batches)



# %%
dataloader = DataLoader(X,y,16)
# for i in dataloader:
#     print (len (i))
    # %%
print(len(dataloader))
#%%
shuffle(dataloader)
# %%
