#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)


# In[14]:


def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size*percentage/100)]

def make_dataset(dataset, n_bus):
    x_raw_1, y_raw_1 = [], []
    x_raw, y_raw = [], []

    for i in range(len(dataset)):
        for n in range(n_bus):
            x_raw_1.extend(dataset[i, 4*n+1:4*n+3])
            y_raw_1.extend(dataset[i, 4*n+3:4*n+5])
        x_raw.append(x_raw_1)
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


# In[27]:


dataset1 = pd.read_excel('dataset\Grid_14 bus_1.xlsx').values
dataset2 = pd.read_excel('dataset\Grid_14 bus_2.xlsx').values


# In[28]:


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


# In[29]:


class My_NN(torch.nn.Module):
    def __init__(self, input_size=None, hidden_size1=None, hidden_size2=None, output_size=None):
        super(My_NN, self).__init__()
        self.input_size = input_size if input_size is not None else 18 
        self.hidden_size1 = hidden_size1 if hidden_size1 is not None else 38
        self.hidden_size2 = hidden_size2 if hidden_size2 is not None else 38
        self.output_size = output_size if output_size is not None else 18
        
        self.lin1 = Linear(self.input_size, self.hidden_size1)
        self.lin2 = Linear(self.hidden_size1, self.hidden_size2)
        self.lin3 = Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        
        x = self.lin1(x)
        x = torch.tanh(x)

        x = self.lin2(x)
        x = torch.tanh(x)

        x = self.lin3(x)

        return x
    
    def save_weights(self, model, name):
        torch.save(model, name)


# In[30]:


get_ipython().run_cell_magic('time', '', '\ninput_size = n_bus*2\nhidden_size1 = 30\nhidden_size2 = 30\noutput_size = n_bus*2\nlr = 0.001\n\nmodel = My_NN(input_size, hidden_size1, hidden_size2, output_size)\noptimizer = torch.optim.Adam(model.parameters(), lr=lr)\ntrain_loss_list, val_loss_list = [], []\n\ncount=0\npatience=10000\nlossMin = 1e10\n\nfor epoch in range(10001):\n\n    model.train()\n    optimizer.zero_grad()\n    y_train_prediction = model(x_norm_train)\n    train_loss = MSE(denormalize_output(y_train_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_train, y_val_mean, y_val_std))\n    train_loss.backward()\n    optimizer.step()\n    train_loss_list.append(train_loss.detach())\n\n    model.eval()\n    y_val_prediction = model(x_norm_val)\n    val_loss = MSE(denormalize_output(y_val_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_val, y_val_mean, y_val_std))\n    val_loss_list.append(val_loss.detach())\n\n    #early stopping\n    if (val_loss < lossMin):\n        lossMin = val_loss\n        count = 0\n        best_epoch = epoch\n        best_train_loss = train_loss\n        best_val_loss = val_loss\n        model.save_weights(model, "[PyG] [14 bus] Best_NN_model.pt")\n    else:\n        count+=1\n        if(count>patience):\n            print("early stop at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))\n            print("best val at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(best_epoch, best_train_loss, best_val_loss))\n            break\n    \n    #if (train_loss <= 0):\n    #    print("min train loss at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))\n    #    break\n\n    if (epoch % 10) == 0:\n        print(\'epoch: {:d}    train loss: {:.7f}    val loss: {:.7f}\'.format(epoch, train_loss, val_loss))\n')


# In[31]:


plt.title('NN on power flow dataset')
plt.plot(train_loss_list, label="train loss")
plt.plot(val_loss_list, label="val loss")
plt.yscale('log')
plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

print('last epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(epoch, train_loss, val_loss))
print('best epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(best_epoch, best_train_loss, best_val_loss))


# In[32]:


for name, param in model.named_parameters():
  print(name)
  print(param.size())

param = sum(p.numel() for p in model.parameters() if p.requires_grad)
param


# In[33]:


model.eval()

y_train_prediction = model(x_norm_train)
train_loss = MSE(denormalize_output(y_train_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_train, y_val_mean, y_val_std))
print("Train output ground-truth: \n" + str(y_raw_train.detach().numpy()[0]))
print("Train output prediction: \n" + str(denormalize_output(y_train_prediction, y_val_mean, y_val_std).detach().numpy()[0]))
print('Train loss (MSE): {:.7f}'.format(train_loss))

print("===========================================================================")

y_val_prediction = model(x_norm_val)
val_loss = MSE(denormalize_output(y_val_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_val, y_val_mean, y_val_std))
print("Train output ground-truth: \n" + str(y_raw_val.detach().numpy()[0]))
print("Train output prediction: \n" + str(denormalize_output(y_val_prediction, y_val_mean, y_val_std).detach().numpy()[0]))
print('Train loss (MSE): {:.7f}'.format(val_loss))


# In[34]:


best_model = torch.load("[PyG] [14 bus] Best_NN_model.pt")
best_model.eval()

y_train_prediction = best_model(x_norm_train)
train_loss = MSE(denormalize_output(y_train_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_train, y_val_mean, y_val_std))
print("Train output ground-truth: \n" + str(y_raw_train.detach().numpy()[0]))
print("Train output prediction: \n" + str(denormalize_output(y_train_prediction, y_val_mean, y_val_std).detach().numpy()[0]))
print('Train loss (MSE): {:.7f}'.format(train_loss))

print("===========================================================================")

y_val_prediction = best_model(x_norm_val)
val_loss = MSE(denormalize_output(y_val_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_val, y_val_mean, y_val_std))
print("Train output ground-truth: \n" + str(y_raw_val.detach().numpy()[0]))
print("Train output prediction: \n" + str(denormalize_output(y_val_prediction, y_val_mean, y_val_std).detach().numpy()[0]))
print('Train loss (MSE): {:.7f}'.format(val_loss))


# In[35]:


get_ipython().run_cell_magic('time', '', '\nbest_model = torch.load("[PyG] [14 bus] Best_NN_model.pt")\nbest_model.eval()\n\ntest_loss_list = []\n\nfor i in range(102):\n    \n    dataset = pd.read_excel(\'dataset\\Grid_14 bus_%d.xlsx\' % (i+1)).values\n    test_percentage = 100\n    test_dataset = slice_dataset(dataset, test_percentage)\n    x_raw_test, y_raw_test = make_dataset(test_dataset, n_bus)\n    x_norm_test, y_norm_test, _, _, _, _ = normalize_dataset(x_raw_test, y_raw_test)\n    \n    print(\'dataset {:d}\'.format(i+1))\n    \n    y_test_prediction = best_model(x_norm_test)\n    test_loss = MSE(denormalize_output(y_test_prediction, y_val_mean, y_val_std), denormalize_output(y_norm_test, y_val_mean, y_val_std))\n    \n    if i == 0:\n        print(\'Train loss (MSE): {:.7f}\'.format(test_loss.detach().numpy()))\n    elif i == 1:\n        print(\'Val loss (MSE): {:.7f}\'.format(test_loss.detach().numpy()))\n    else:\n        print(\'Test loss (MSE): {:.7f}\'.format(test_loss))\n        test_loss_list.append(test_loss.detach().numpy())\n    \n    print("===========================")\n\ncolumn = []\nfor i in range(100):\n    column.append(\'test loss %d\' % (i+1))\n    \ntest_loss_file = pd.DataFrame([test_loss_list], columns=column)\ntest_loss_file.to_excel("[PyG] [14 bus] [MSE] NN test loss.xlsx")\nprint("\\ntest loss file saved!\\n")\n')


# In[36]:


get_ipython().run_cell_magic('time', '', '\nbest_model = torch.load("[PyG] [14 bus] Best_NN_model.pt")\nbest_model.eval()\n\ntest_loss_list = []\n\nfor i in range(102):\n    \n    dataset = pd.read_excel(\'dataset\\Grid_14 bus_%d.xlsx\' % (i+1)).values\n    test_percentage = 100\n    test_dataset = slice_dataset(dataset, test_percentage)\n    x_raw_test, y_raw_test = make_dataset(test_dataset, n_bus)\n    x_norm_test, y_norm_test, _, _, _, _ = normalize_dataset(x_raw_test, y_raw_test)\n    \n    print(\'dataset {:d}\'.format(i+1))\n    \n    yhat = denormalize_output(best_model(x_norm_test), y_val_mean, y_val_std)\n    y = y_raw_test\n    test_loss_NRMSE = NRMSE(yhat, y)\n    \n    if i == 0:\n        print(\'Train loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n    elif i == 1:\n        print(\'Val loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n    else:\n        print(\'Test loss (NRMSE): {:.7f}\'.format(test_loss_NRMSE.detach().numpy()))\n        test_loss_list.append(test_loss_NRMSE.detach().numpy())\n    \n    print("===========================")\n\ncolumn = []\nfor i in range(100):\n    column.append(\'test loss %d\' % (i+1))\n    \ntest_loss_file = pd.DataFrame([test_loss_list], columns=column)\ntest_loss_file.to_excel("[PyG] [14 bus] [NRMSE] NN test loss.xlsx")\nprint("\\ntest loss file saved!\\n")\n')

