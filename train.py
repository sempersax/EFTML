import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import preshower2 as ps
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot
from regressor import Net
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader

matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# defining the model
model = Net()
model = model.double()
print(model)


# Loading data from preshower


fourVectors, rhatcs, wilsonCos = ps.train_data()

X_data = np.array([fourVectors[key] for key in fourVectors.keys()])
X_data = np.transpose(X_data)


Y_data = np.array([rhatcs[key] for key in rhatcs.keys()])
Y_data = np.transpose(Y_data)

trainloader = torch.utils.data.DataLoader((X_data,Y_data), batch_size=10, shuffle=True, num_workers=1)

#ts1 = 0.2 means 20% of data will be validation data
ts1 = 0.2
rs1 = 42
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = ts1, random_state = rs1)

# splitting the output by column (by rhatc)
Y_train = np.array(np.hsplit(Y_train, len(Y_train[0])))
Y_test = np.array(np.hsplit(Y_test, len(Y_test[0])))

# converting data to the correct shape                                                                                                                                                                                                                                                            
#X_train = X_train.reshape(X_train.shape[0], 1).astype('float64')
X_train = torch.from_numpy(X_train)
#X_test = X_test.reshape(X_test.shape[0], 1).astype('float64')
X_test = torch.from_numpy(X_test)

#Y_train = Y_train.reshape(Y_train.shape[0], 1).astype('float64')
Y_train = torch.from_numpy(Y_train)
#Y_test = Y_test.reshape(Y_test.shape[0], 1).astype('float64')
Y_test = torch.from_numpy(Y_test)

print("Training sample : "+str(X_train.shape)+" , Validation sample : "+str(X_test.shape))
print("Training output : "+str(Y_train.shape)+" , Validation output : "+str(Y_test.shape))

""" 
Here we create a list for the training data.  For each rhatc (rhat1, ..., rhat6,...)
we create a dataset with the same inputs.  For 2 wilson coefficients, this results in
6 datasets.  The same is done for the test data.
"""
train_data = [TensorDataset(X_train, data) for data in Y_train]
test_data = [TensorDataset(X_test, data) for data in Y_test]

"""
Here we do the dataloader thing, for each different rhatc
"""
trainers = [DataLoader(dataset = _, batch_size = 128, shuffle = True) for _ in train_data]
testers = [DataLoader(dataset = _, batch_size=1, shuffle=False) for _ in test_data]


model.to(device)



def train_model(model, epochs, lr, trainloader, testloader):
    """
    Defining the loss function, MSE -> Mean Squared Error
    which I think is what RASCAL uses
    """
#   loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        model.train() # train mode (affects batchnorm layers:
                      # in the subsequent forward passes they'll
                      # exhibit 'train' behaviour, i.e. they'll
                      # normalize activations over batches)
        train_epoch_losses = []
        for i, (X, y) in enumerate(tqdm(trainloader)):
            #X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_losses.append(loss.item())

        train_loss.append(np.mean(train_epoch_losses))
        model.eval() # test mode (affects batchnorm layers:
                     # in the subsequent forward passes they'll
                     # exhibit 'test' behaviour, i.e. they'll
                     # use the accumulated running statistics
                     # to normalize activations)
        epoch_losses = []

        with torch.no_grad(): # avoid calculating gradients during evaluation
            for X, y in testloader:

                pred = model(X)

                epoch_losses.append(loss_fn(pred, y).item())
                #_, pred = torch.max(pred.data, 1) # pred = index of maximal output along axis=1
                #epoch_accuracies.append(
                #    (pred == y).to(torch.float32).mean().item()
                #)
        test_loss.append(np.mean(epoch_losses))
        if epoch % 10 == 0:
            print(f"For epoch {epoch}: \n", "Epoch Training loss = "+str(np.mean(train_epoch_losses))+", Validation loss = "+str(np.mean(epoch_losses)))


    return dict(
        train_loss=train_loss,
        test_loss=test_loss,
    )

if __name__ == "__main__":
    
    epochs = 30
    lr = 10**-3
    results = []
    for i in range(len(trainers)):
        print(f"rhatc{i+1} being trained")
        result = train_model(model, epochs, lr, trainers[i], testers[i])
        results.append(result)
        print('\n\n\n')
    Path = os.getcwd()+"/regression.pt"
    torch.save(model.state_dict(), Path)
    
    for i in range(len(results)):
        if len(results[i]['train_loss'])==len(results[i]['test_loss']):
            plt.plot(results[i]['train_loss'], label='Training loss')
            plt.plot(results[i]['test_loss'], label='Validation loss')
            plt.legend()
            plt.ylim(-0.1*max(results[i]['train_loss']),1.1*max(results[i]['train_loss']))
            plt.savefig(os.getcwd()+f"/regress_loss{i+1}.pdf")
            plt.close()
        else:
            print("loss length is not the same")
