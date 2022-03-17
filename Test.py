#!/usr/bin/env python
# coding: utf-8

# ## **Load Libraries**

# In[ ]:


import os
import tqdm
import shutil
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from tensorflow.keras.utils import to_categorical


# In[ ]:


import matplotlib
matplotlib.use('agg') # Use Agg Backend For Matplotlib


# ## **Mount Drive**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive') # Mounts Drive In Content Directory (Under Files Section)


# In[ ]:


Path = r'/content/drive/MyDrive/ECG ML'


# ## **Load Dataset**

# In[ ]:


# Load Data List From Pickle Format
pickle_in = open(os.path.join(Path, 'Dataset', 'Training_Data.pickle'), 'rb')
Training_Data = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join(Path, 'Dataset', 'Validation_Data.pickle'), 'rb')
Validation_Data = pickle.load(pickle_in)
pickle_in.close()


# In[ ]:


# Load Test Data From TXT or CSV Files
Test_Data = []

Dict = {'nan': -1, 'Normal': 0, 'LBBB': 1, 'RBBB': 2, 'PVC': 3}
main_df = pd.read_excel(os.path.join(Path, 'Test', 'Reference.xlsx'), index_col = 'Recording').sort_index()

for Folder_Name in main_df.index:
  Label = str(main_df.loc[Folder_Name, 'Label'])
  Files = os.listdir(os.path.join(Path, 'Test', Folder_Name))
  
  if len(Files) == 12:
    for index, File_Name in enumerate(sorted(Files)):
      Data = np.genfromtxt(os.path.join(Path, 'Test', Folder_Name, File_Name), dtype = np.float64)

      # Drop Excess Data Or Fill Up Missing Data
      Length = len(Data)
      if Length > 2800:
        Data = Data[:2800]
      elif Length < 2800:
        for _ in range(2800 - Length):
          Data = np.append(Data, Data[-1])

      if index == 0:
        File = Data 
      else:  
        File = np.column_stack((File, Data))
    
    Test_Data.append([File, Dict[Label]])

  elif len(Files) == 1:
    Data = np.genfromtxt(os.path.join(Path, 'Test', Folder_Name, Files[0]), dtype = np.float64, delimiter = ',', skip_header = 1)

    # Drop Excess Data Or Fill Up Missing Data
    Length = len(Data)
    if Length > 2800:
      Data = Data[:2800]
    elif Length < 2800:
      for _ in range(2800 - Length):
        Data = np.vstack((Data, Data[-1]))

    Test_Data.append([Data, Dict[Label]])


# ## **Initialize TPU**

# In[ ]:


# Standard Syntax Code To Intialize TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


# ## **Load Best Accuracy Model**

# In[ ]:


# Loads Model
model = load_model(os.path.join(Path, 'Model', 'best_val_acc.h5'))


# ## **Accuracy on Best Accuracy Model**

# In[ ]:


# Separate Target
x, y = np.array([Sample[0] for Sample in Test_Data], dtype = np.float64), np.array([Sample[1] for Sample in Test_Data], dtype = np.int64)

# Check If All Labels Are Present, If So Evaluate Model
if -1 not in y:
  y = to_categorical(y)

  loss, acc = model.evaluate(x, y)
  print('Loss on Test Data : ', loss)
  print('Accuracy on Test Data :', '{:.4%}'.format(acc))
else:
  print('True labels absent')


# ## **Load Best Loss Model**

# In[ ]:


# Loads Model
model = load_model(os.path.join(Path, 'Model', 'best_val_loss.h5'))


# ## **Accuracy on Best Loss Model**

# In[ ]:


x, y = np.array([Sample[0] for Sample in Test_Data], dtype = np.float64), np.array([Sample[1] for Sample in Test_Data], dtype = np.int64)

# Check If All Labels Are Present, If So Evaluate Model
if -1 not in y:
  y = to_categorical(y)

  loss, acc = model.evaluate(x, y)
  print('Loss on Test Data : ', loss)
  print('Accuracy on Test Data :', '{:.4%}'.format(acc))
else:
  print('True labels absent')


# ## **Figures**

# In[ ]:


plt.style.use('seaborn') # Style To Use For Matplotlib Plots
 
Time = [i / 500 for i in range(2800)]
Dict = {-1: 'nan', 0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
Labels = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Test Figures (Lead II)')): # Checks If Directory Exist
    shutil.rmtree(os.path.join(Path, 'Test Figures (Lead II)')) # Removes Directory
os.mkdir(os.path.join(Path, 'Test Figures (Lead II)')) # Creates Directory

Count = 0    
for Sample, Label in tqdm.tqdm(Test_Data, unit_scale = True, miniters = 1, desc = 'Plotting Test Data (Lead II) '): # Progress Bar
    df = pd.DataFrame(Sample)

    # Plot And Save
    fig, ax = plt.subplots(1)
    fig.set_size_inches(37.33, 11.5)

    ax.plot(Time, df.iloc[:, 1], 'b', label = Labels[1])
    ax.legend(loc = 'upper right', prop = {'size': 18})
    ax.set_xlabel('Time', fontsize = 18)
    ax.set_ylabel('mV', fontsize = 18)  
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis = 'both', colors = 'black', labelsize = 16)

    fig.tight_layout()
    fig.suptitle(Dict[Label], fontsize = 22)
    fig.subplots_adjust(top = 0.96, right = 0.978)
    fig.savefig(os.path.join(Path, 'Test Figures (Lead II)', f'Figure {Count}.png'), dpi = 300)
    plt.close(fig)
 
    Count += 1


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Test Figures')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Test Figures')) # Removes Directory
os.mkdir(os.path.join(Path, 'Test Figures')) # Creates Directory

Count = 0
for Sample, Label in tqdm.tqdm(Test_Data, unit_scale = True, miniters = 1, desc = 'Plotting Test Data '):
  df = pd.DataFrame(Sample)
  
  # Plot And Save
  fig, ax = plt.subplots(6, 2, sharex = True)
  fig.set_size_inches(37.33, 21)
  for i in range(2):
    for j in range(6):
      ax[j][i].plot(Time, df.iloc[:, i * 6 + j], 'b', label = Labels[i * 6 + j])
      ax[j][i].legend(loc = 'upper right', prop = {"size": 14})
      ax[j][i].set_ylabel('mV', fontsize = 12)  
      ax[j][i].yaxis.label.set_color('black')
      ax[j][i].tick_params(axis = 'both', colors = 'black')
    ax[j][i].set_xlabel('Time', fontsize = 12)
    ax[j][i].xaxis.label.set_color('black')
  fig.tight_layout()
  fig.suptitle(Dict[Label], fontsize = 18)
  fig.subplots_adjust(top = 0.96, right = 0.978)
  fig.savefig(os.path.join(Path, 'Test Figures', f'{main_df.index[Count]}.png'), dpi = 300)
  plt.close(fig)

  Count += 1


# ## **Predictions**

# In[ ]:


Dict = {-1: 'nan', 0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}

# Separate Target Variable
x, y = np.array([Sample[0] for Sample in Test_Data], dtype = np.float64), np.array([Sample[1] for Sample in Test_Data], dtype = np.int64)

# Take Predictions
y_true = [Dict[Label] for Label in y]
y_pred = [Dict[Label] for Label in np.argmax(model.predict(x), axis = 1)]

# Push Results To CSV File
Results = pd.DataFrame(list(zip(main_df.index, y_true, y_pred)), columns = ['Name', 'Actual', 'Prediction'])
Results.to_csv(os.path.join(Path, 'Test Results.csv'), index = False)


# In[ ]:


Results


# ## **Single Training Data File**

# In[ ]:


Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}

Class = random.randint(0, 3) # [0, 3] Included, Class to Use
Sample = random.randint(0, len(Training_Data[Class]) - 1) # [0, len(Training_Data[Class]) - 1] Included, Sample to Use

# Take Prediction
print('Prediction:', Dict[np.argmax(model.predict(np.array([Training_Data[Class][Sample]])))])
print('Actual:', Dict[Class])


# ## **Single Test Data File**

# In[ ]:


Folder_Name = 'A0001' # Enter Folder Name Here
True_Label = 'RBBB' # Enter True Label Here ('nan' If Not Known)

Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
Files = os.listdir(os.path.join(Path, 'Test', Folder_Name))

if len(Files) == 12:
  for index, File_Name in enumerate(sorted(Files)):
    Data = np.genfromtxt(os.path.join(Path, 'Test', Folder_Name, File_Name), dtype = np.float64)

    # Drop Excess Data Or Fill Up Missing Data
    Length = len(Data)
    if Length > 2800:
      Data = Data[:2800]
    elif Length < 2800:
      for _ in range(2800 - Length):
        Data = np.append(Data, Data[-1])

    if index == 0:
      File = Data 
    else:  
      File = np.column_stack((File, Data))

elif len(Files) == 1:
  Data = np.genfromtxt(os.path.join(Path, 'Test', Folder_Name, Files[0]), dtype = np.float64, delimiter = ',', skip_header = 1)

  # Drop Excess Data Or Fill Up Missing Data
  Length = len(Data)
  if Length >= 2800:
    Data = Data[:2800]
  elif Length < 2800:
    for _ in range(2800 - Length):
      Data = np.vstack((Data, Data[-1]))
  
  File = Data.copy()

# Take Prediction
print('Prediction:', Dict[np.argmax(model.predict(np.array([File])))])
print('Actual:', True_Label)


# ## **Lead DII Stacked Plots**

# In[ ]:


plt.style.use('seaborn') # Style To Use For Matplotlib Plots
 
Time = [i / 500 for i in range(2800)]
Dict = {-1: 'nan', 0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
Labels = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Stacked Lead II Figures')): # Checks If Directory Exist
    shutil.rmtree(os.path.join(Path, 'Stacked Lead II Figures')) # Removes Directory
os.mkdir(os.path.join(Path, 'Stacked Lead II Figures')) # Creates Directory

Count = 0
for Test_File, Label in Test_Data:
  Sample = random.randint(0, len(Training_Data[Label]) - 1)
  Train_File = Training_Data[Label][Sample]

  test_df = pd.DataFrame(Test_File)
  train_df = pd.DataFrame(Train_File)

  # Plot And Save
  fig, ax = plt.subplots(2, sharex = True)
  fig.set_size_inches(37.33, 21)

  ax[0].plot(Time, train_df.iloc[:, 1], 'b', label = 'Train Lead II')
  ax[1].plot(Time, test_df.iloc[:, 1], 'b', label = 'Test Lead II')

  for i in range(2):
    ax[i].legend(loc = 'upper right', prop = {'size': 18})
    ax[i].set_ylabel('mV', fontsize = 18)  
    ax[i].yaxis.label.set_color('black')
    ax[i].tick_params(axis = 'both', colors = 'black', labelsize = 16)
  ax[i].set_xlabel('Time', fontsize = 18)
  ax[i].xaxis.label.set_color('black')

  fig.tight_layout()
  fig.suptitle(Dict[Label], fontsize = 22)
  fig.subplots_adjust(top = 0.96, right = 0.978)
  fig.savefig(os.path.join(Path, 'Stacked Lead II Figures', f'Figure {Count}.png'), dpi = 300)
  plt.close(fig)

  Count += 1

