#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)


# In[2]:


def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size*percentage/100)]

def make_dataset(dataset, n_bus):
    x_raw_1, y_raw_1 = [], []
    x_raw, y_raw = [], []

    for i in range(len(dataset)):
        for n in range(n_bus):
            x_raw_1.append(list([dataset[i, 4*n+1], dataset[i, 4*n+2]]))
            y_raw_1.extend(dataset[i, 4*n+3:4*n+5])

        x_raw.append(list(x_raw_1))
        y_raw.append(y_raw_1)
        x_raw_1, y_raw_1 = [], []

    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    return x_raw, y_raw

def normalize_dataset(x, y):
    x_mean = torch.mean(x,0)
    y_mean = torch.mean(y,0)
    x_std = torch.std(x,0)
    y_std = torch.std(y,0)
    x_norm = (x-x_mean)/x_std
    y_norm = (y-y_mean)/y_std
    x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
    y_norm = torch.where(torch.isnan(y_norm), torch.zeros_like(y_norm), y_norm)
    x_norm = torch.where(torch.isinf(x_norm), torch.zeros_like(x_norm), x_norm)
    y_norm = torch.where(torch.isinf(y_norm), torch.zeros_like(y_norm), y_norm)
    return x_norm, y_norm, x_mean, y_mean, x_std, y_std

def denormalize_output(y_norm, y_mean, y_std):
    y = y_norm*y_std+y_mean
    return y

def NRMSE(yhat,y):
    return torch.sqrt(torch.mean(((yhat-y)/torch.std(yhat,0))**2))

def MSE(yhat,y):
    return torch.mean((yhat-y)**2)


# In[3]:


dataset1 = pd.read_excel('dataset\Grid_14 bus_1.xlsx').values
dataset2 = pd.read_excel('dataset\Grid_14 bus_2.xlsx').values


# In[4]:


train_percentage = 100
val_percentage = 100

train_dataset = slice_dataset(dataset1, train_percentage)
val_dataset = slice_dataset(dataset2, val_percentage)

n_bus = 14

#actual data
x_raw_train, y_raw_train = make_dataset(train_dataset, n_bus)
x_raw_val, y_raw_val = make_dataset(val_dataset, n_bus)

#normalized data
x_norm_train, y_norm_train, _, _, _, _ = normalize_dataset(x_raw_train, y_raw_train)
x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val)


# In[5]:


x_train, y_train = x_norm_train, y_norm_train
x_val, y_val = x_norm_val, y_norm_val
edge_index = torch.tensor([[0, 1, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 4, 7, 5, 8, 5, 9, 1, 10, 10, 11, 11, 12, 11, 13],
                           [1, 0, 2, 1, 3, 1, 4, 2, 5, 3, 6, 4, 7, 4, 8, 5, 9, 5, 10, 1, 11, 10, 12, 11, 13, 11]], dtype=torch.long)

data_train_list, data_val_list = [], []
for i,_ in enumerate(x_train):
    data_train_list.append(Data(x=x_train[i], y=y_train[i], edge_index=edge_index))
for i,_ in enumerate(x_val):
    data_val_list.append(Data(x=x_val[i], y=y_val[i], edge_index=edge_index))

train_loader = DataLoader(data_train_list, batch_size=1)
val_loader = DataLoader(data_val_list, batch_size=1)


# In[6]:


class My_GNN_NN(torch.nn.Module):
    def __init__(self, node_size=None, feat_in=None, feat_size1=None, hidden_size1=None, output_size=None):
        super(My_GNN_NN, self).__init__()
        self.feat_in = feat_in if feat_in is not None else 2
        self.feat_size1 = feat_in if feat_in is not None else 4
        self.hidden_size1 = hidden_size1 if hidden_size1 is not None else 20
        self.output_size = output_size if output_size is not None else 12
        
        self.conv1 = GCNConv(feat_in, feat_size1)
        self.lin1 = Linear(node_size*feat_size1, hidden_size1)
        self.lin2 = Linear(hidden_size1, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.tanh(x)

        x = x.flatten(start_dim = 0)
        x = self.lin1(x)
        x = torch.tanh(x)

        x = self.lin2(x)

        return x
    
    def save_weights(self, model, name):
        torch.save(model, name)


# In[7]:


feat_in = 2
feat_size1 = 4
hidden_size1 = 30
output_size = n_bus*2
lr = 0.0001

model = My_GNN_NN(n_bus, feat_in, feat_size1, hidden_size1, output_size)
for name, param in model.named_parameters():
  print(name)
  print(param.size())

param = sum(p.numel() for p in model.parameters() if p.requires_grad)
param


# In[8]:


get_ipython().run_cell_magic('time', '', '\nfeat_in = 2\nfeat_size1 = 4\nhidden_size1 = 30\noutput_size = n_bus*2\nlr = 0.0001\n\nmodel = My_GNN_NN(n_bus, feat_in, feat_size1, hidden_size1, output_size)\noptimizer = torch.optim.Adam(model.parameters(), lr=lr)\ntrain_loss_list, val_loss_list = [], []\n\ncount=0\npatience=2000\nlossMin = 1e10\n\nfor epoch in range(2001):\n\n    model.train()\n    train_loss = 0\n    for batch in train_loader:\n        optimizer.zero_grad()\n        y_train_prediction = model(batch)\n        loss = MSE(denormalize_output(y_train_prediction, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))\n        loss.backward()\n        optimizer.step()\n        train_loss += loss.item() * batch.num_graphs\n    train_loss /= len(train_loader.dataset)\n    train_loss_list.append(train_loss)\n\n    model.eval()\n    val_loss=0\n    for batch in val_loader:\n        y_val_prediction = model(batch)\n        loss = MSE(denormalize_output(y_val_prediction, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))\n        val_loss += loss.item() * batch.num_graphs\n    val_loss /= len(val_loader.dataset)\n    val_loss_list.append(val_loss)\n\n    #early stopping\n    if (val_loss < lossMin):\n        lossMin = val_loss\n        count = 0\n        best_epoch = epoch\n        best_train_loss = train_loss\n        best_val_loss = val_loss\n        model.save_weights(model, "[PyG] [14 bus] Best_GNN_NN_model.pt")\n    else:\n        count+=1\n        if(count>patience):\n            print("early stop at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))\n            print("best val at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(best_epoch, best_train_loss, best_val_loss))\n            break\n    \n    if (train_loss <= 0):\n        print("min train loss at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))\n        break\n\n    if (epoch % 10) == 0:\n        print(\'epoch: {:d}    train loss: {:.7f}    val loss: {:.7f}\'.format(epoch, train_loss, val_loss))\n')


# In[9]:


plt.title('GNN NN on power flow dataset')
plt.plot(train_loss_list, label="train loss")
plt.plot(val_loss_list, label="val loss")
plt.yscale('log')
plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

print('last epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(epoch, train_loss, val_loss))
print('best epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(best_epoch, best_train_loss, best_val_loss))


# In[10]:


model.eval()

y_train_prediction_1 = model(train_loader.dataset[0])
train_loss_1 = MSE(denormalize_output(y_train_prediction_1, y_val_mean, y_val_std), denormalize_output(y_norm_train[0], y_val_mean, y_val_std))
print("[1 datapoint] Train output ground-truth: \n" + str(y_raw_train[0].detach().numpy()))
print("[1 datapoint] Train output prediction: \n" + str(denormalize_output(y_train_prediction_1, y_val_mean, y_val_std).detach().numpy()))
print('[1 datapoint] Train loss (MSE): {:.7f}'.format(train_loss_1))

train_loss = 0
for batch in train_loader:
    pred = model(batch)
    loss = MSE(denormalize_output(pred, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))
    train_loss += loss.item() * batch.num_graphs
train_loss /= len(train_loader.dataset)
print('Train loss (MSE): {:.7f}'.format(train_loss))

print("=========================================================================")

y_val_prediction_1 = model(val_loader.dataset[0])
val_loss_1 = MSE(denormalize_output(y_val_prediction_1, y_val_mean, y_val_std), denormalize_output(y_norm_val[0], y_val_mean, y_val_std))
print("[1 datapoint] Val output ground-truth: \n" + str(y_raw_val[0].detach().numpy()))
print("[1 datapoint] Val output prediction: \n" + str(denormalize_output(y_val_prediction_1, y_val_mean, y_val_std).detach().numpy()))
print('[1 datapoint] Val loss (MSE): {:.7f}'.format(val_loss_1))

val_loss=0
for batch in val_loader:
    pred = model(batch)
    loss = MSE(denormalize_output(pred, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))
    val_loss += loss.item() * batch.num_graphs
val_loss /= len(val_loader.dataset)
print('Val loss (MSE): {:.7f}'.format(val_loss))


# In[11]:


best_model = torch.load("[PyG] [14 bus] Best_GNN_NN_model.pt")
best_model.eval()

y_train_prediction_1 = best_model(train_loader.dataset[0])
train_loss_1 = MSE(denormalize_output(y_train_prediction_1, y_val_mean, y_val_std), denormalize_output(y_norm_train[0], y_val_mean, y_val_std))
print("[1 datapoint] Train output ground-truth: \n" + str(y_raw_train[0].detach().numpy()))
print("[1 datapoint] Train output prediction: \n" + str(denormalize_output(y_train_prediction_1, y_val_mean, y_val_std).detach().numpy()))
print('[1 datapoint] Train loss (MSE): {:.7f}'.format(train_loss_1))

train_loss = 0
for batch in train_loader:
    pred = best_model(batch)
    loss = MSE(denormalize_output(pred, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))
    train_loss += loss.item() * batch.num_graphs
train_loss /= len(train_loader.dataset)
print('Train loss (MSE): {:.7f}'.format(train_loss))

print("=========================================================================")

y_val_prediction_1 = best_model(val_loader.dataset[0])
val_loss_1 = MSE(denormalize_output(y_val_prediction_1, y_val_mean, y_val_std), denormalize_output(y_norm_val[0], y_val_mean, y_val_std))
print("[1 datapoint] Val output ground-truth: \n" + str(y_raw_val[0].detach().numpy()))
print("[1 datapoint] Val output prediction: \n" + str(denormalize_output(y_val_prediction_1, y_val_mean, y_val_std).detach().numpy()))
print('[1 datapoint] Val loss (MSE): {:.7f}'.format(val_loss_1))

val_loss=0
for batch in val_loader:
    pred = best_model(batch)
    loss = MSE(denormalize_output(pred, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))
    val_loss += loss.item() * batch.num_graphs
val_loss /= len(val_loader.dataset)
print('Val loss (MSE): {:.7f}'.format(val_loss))


# In[12]:


get_ipython().run_cell_magic('time', '', '\nbest_model = torch.load("[PyG] [14 bus] Best_GNN_NN_model.pt")\nbest_model.eval()\n\ntest_loss_list = []\n\nfor i in range(102):\n    \n    dataset = pd.read_excel(\'dataset\\Grid_14 bus_%d.xlsx\' % (i+1)).values\n    test_percentage = 100\n    test_dataset = slice_dataset(dataset, test_percentage)\n    x_raw_test, y_raw_test = make_dataset(test_dataset, n_bus)\n    x_norm_test, y_norm_test, _, _, _, _ = normalize_dataset(x_raw_test, y_raw_test)\n    \n    x_test, y_test = x_norm_test, y_norm_test\n    \n    data_test_list = []\n    for j,_ in enumerate(x_test):\n        data_test_list.append(Data(x=x_test[j], y=y_test[j], edge_index=edge_index))\n\n    test_loader = DataLoader(data_test_list, batch_size=1)\n    \n    print(\'dataset {:d}\'.format(i+1))\n    \n    test_loss = 0\n    for batch in test_loader:\n        y_test_prediction = best_model(batch)\n        loss = MSE(denormalize_output(y_test_prediction, y_val_mean, y_val_std), denormalize_output(batch.y, y_val_mean, y_val_std))\n        test_loss += loss.item() * batch.num_graphs\n    test_loss /= len(test_loader.dataset)\n    \n    if i == 0:\n        print(\'Train loss (MSE): {:.7f}\'.format(test_loss))\n    elif i == 1:\n        print(\'Val loss (MSE): {:.7f}\'.format(test_loss))\n    else:\n        print(\'Test loss (MSE): {:.7f}\'.format(test_loss))\n        test_loss_list.append(test_loss)\n    \n    print("===========================")\n\ncolumn = []\nfor i in range(100):\n    column.append(\'test loss %d\' % (i+1))\n    \ntest_loss_file = pd.DataFrame([test_loss_list], columns=column)\ntest_loss_file.to_excel("[PyG] [14 bus] [MSE] GNN NN test loss.xlsx")\nprint("\\ntest loss file saved!\\n")\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nbest_model = torch.load("[PyG] [14 bus] Best_GNN_NN_model.pt")\nbest_model.eval()\n\ntest_loss_list = []\n\nfor i in range(102):\n    \n    dataset = pd.read_excel(\'dataset\\Grid_14 bus_%d.xlsx\' % (i+1)).values\n    test_percentage = 100\n    test_dataset = slice_dataset(dataset, test_percentage)\n    x_raw_test, y_raw_test = make_dataset(test_dataset, n_bus)\n    x_norm_test, y_norm_test, _, _, _, _ = normalize_dataset(x_raw_test, y_raw_test)\n    \n    x_test, y_test = x_norm_test, y_norm_test\n    \n    data_test_list = []\n    for j,_ in enumerate(x_test):\n        data_test_list.append(Data(x=x_test[j], y=y_test[j], edge_index=edge_index))\n\n    test_loader = DataLoader(data_test_list, batch_size=1)\n    \n    print(\'dataset {:d}\'.format(i+1))\n    \n    test_loss = 0\n    yhat = torch.empty(0, n_bus*2)\n    for batch in test_loader:\n        y_test_prediction = best_model(batch)\n        yhat = torch.cat((yhat, y_test_prediction.reshape(1, n_bus*2)))\n    \n    yhat = denormalize_output(yhat, y_val_mean, y_val_std)\n    y = y_raw_test\n    test_loss_NRMSE = NRMSE(yhat, y)\n    \n    if i == 0:\n        print(\'Train loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n    elif i == 1:\n        print(\'Val loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n    else:\n        print(\'Test loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n        test_loss_list.append(test_loss_NRMSE.detach().numpy())\n    \n    print("===========================")\n\ncolumn = []\nfor i in range(100):\n    column.append(\'test loss %d\' % (i+1))\n    \ntest_loss_file = pd.DataFrame([test_loss_list], columns=column)\ntest_loss_file.to_excel("[PyG] [14 bus] [NRMSE] GNN NN test loss.xlsx")\nprint("\\ntest loss file saved!\\n")\n')

