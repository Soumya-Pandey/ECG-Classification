#!/usr/bin/env python
# coding: utf-8

# ## **Load Libraries**

# In[ ]:


import os
import tqdm
import shutil
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
 
from IPython.display import display
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense


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
pickle_in = open(os.path.join(Path, 'Dataset', 'x_train.pickle'), 'rb')
x_train = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join(Path, 'Dataset', 'y_train.pickle'), 'rb')
y_train = pickle.load(pickle_in)
pickle_in.close()


# In[ ]:


# Load Data List From Pickle Format
pickle_in = open(os.path.join(Path, 'Dataset', 'x_validation.pickle'), 'rb')
x_validation = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join(Path, 'Dataset', 'y_validation.pickle'), 'rb')
y_validation = pickle.load(pickle_in)
pickle_in.close()


# In[ ]:


# Load Data List From Pickle Format
pickle_in = open(os.path.join(Path, 'Dataset', 'Training_Data.pickle'), 'rb')
Training_Data = pickle.load(pickle_in)
pickle_in.close()
 
pickle_in = open(os.path.join(Path, 'Dataset', 'Validation_Data.pickle'), 'rb')
Validation_Data = pickle.load(pickle_in)
pickle_in.close()


# ## **Initialize TPU**

# In[ ]:


# Standard Syntax Code To Intialize TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


# ## **Residual Unit**

# In[ ]:


# Residual Unit (Layer Using Conv1D)
class ResidualUnit(object):
    '''Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu' '''
 
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer = 'he_normal',
                 dropout_rate = 0.8, kernel_size = 17, preactivation = True,
                 postactivation_bn = False, activation_function = 'relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
 
    def _skip_connection(self, y, downsample, n_filters_in):
        '''Implement skip connection.'''
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides = downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError('Number of samples should always decrease.')
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias = False, kernel_initializer = self.kernel_initializer)(y)
        return y
 
    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center = False, scale = False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x
 
    def __call__(self, inputs):
        '''Residual unit.'''
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding = 'same',
                   use_bias = False, kernel_initializer = self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
 
        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides = downsample,
                   padding = 'same', use_bias = False,
                   kernel_initializer = self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


# ## **Model Architecture**

# In[ ]:


with strategy.scope():
  signal = Input(shape = (2800, 12), dtype = np.float64, name = 'signal') # Input Layer
    
  x = signal

  # 1st Conv1D Layer
  x = Conv1D(64, 16, padding = 'same', use_bias = False,
            kernel_initializer = 'he_uniform')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  # 4 Residual Layers
  x, y = ResidualUnit(1024, 128, kernel_initializer = 'he_uniform',
                      kernel_size = 32, dropout_rate = 0.8)([x, x])
  x, y = ResidualUnit(256, 196, kernel_initializer = 'he_uniform',
                      kernel_size = 32, dropout_rate = 0.8)([x, y])
  x, y = ResidualUnit(64, 256, kernel_initializer = 'he_uniform',
                      kernel_size = 16, dropout_rate = 0.8)([x, y])
  x, _ = ResidualUnit(16, 320, kernel_initializer = 'he_uniform',
                      kernel_size = 16, dropout_rate = 0.8)([x, y])

  # Flatten Layer
  x = Flatten()(x)

  # 2 Dense Layers
  x = Dense(128, activation = 'relu', kernel_initializer = 'he_normal', kernel_regularizer = l2(0.4))(x)
  x = Dense(64, activation = 'sigmoid', kernel_initializer = 'he_normal', kernel_regularizer = l2(0.2))(x)

  # Output Layer
  diagn = Dense(4, activation = 'softmax')(x)

  # Create Model
  model = Model(signal, diagn)


# ## **Model Summary**

# In[ ]:


# Print Model Summary (Graph With Shape)
model.summary()


# ## **Callbacks, Hyperparameters and Model Compilation**

# In[ ]:


if os.path.isdir(os.path.join(Path, 'Model')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Model')) # Removes Directory
os.mkdir(os.path.join(Path, 'Model')) # Creates Directory

# Add Different CallBacks To Model
callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                              patience = 5, min_lr = 1e-3),
             CSVLogger(os.path.join(Path, 'Model', 'training.log'), append = True),
             ModelCheckpoint(os.path.join(Path, 'Model', 'backup_last_model.h5')),
             ModelCheckpoint(os.path.join(Path, 'Model', 'best_val_acc.h5'), monitor = 'val_accuracy', save_best_only = True),
             ModelCheckpoint(os.path.join(Path, 'Model', 'best_val_loss.h5'), monitor = 'val_loss', save_best_only = True)]

# Define Some Parameters (Will Be Used While Training)
batch_size = 64
steps_per_epoch = int(x_train.shape[0] / batch_size)
validation_steps = int(x_validation.shape[0] / batch_size)

with strategy.scope():
  # Compile The Model With Passing Optimizer and Loss Function
  model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-3, decay = 1e-6), metrics = ['accuracy'], steps_per_execution = steps_per_epoch)


# ## **Train Model**

# In[ ]:


# Training Model
model.fit(x_train, y_train, initial_epoch = 0, epochs = 100, batch_size = batch_size, validation_batch_size = batch_size, 
          validation_data = (x_validation, y_validation), steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, callbacks = callbacks)


# ## **Load Best Accuracy Model**

# In[ ]:


# Loads Model
model = load_model(os.path.join(Path, 'Model', 'best_val_acc.h5'))


# ## **Accuracy on Best Accuracy Model**

# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Training Data : ', loss)
print('Accuracy on Training Data :', '{:.4%}'.format(acc))


# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature[:100]]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Training Data (100 Samples of Each Class) : ', loss)
print('Accuracy on Training Data (100 Samples of Each Class) :', '{:.4%}'.format(acc))


# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Validation_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))


# ## **Load Best Loss Model**

# In[ ]:


# Loads Model
model = load_model(os.path.join(Path, 'Model', 'best_val_loss.h5'))


# ## **Accuracy on Best Loss Model**

# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Training Data : ', loss)
print('Accuracy on Training Data :', '{:.4%}'.format(acc))


# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature[:100]]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Training Data (100 Samples of Each Class) : ', loss)
print('Accuracy on Training Data (100 Samples of Each Class) :', '{:.4%}'.format(acc))


# In[ ]:


# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Validation_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Evaluate Model (Accuracy And Loss)
loss, acc = model.evaluate(x, y)
print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))


# ## **Figures**

# In[ ]:


plt.style.use('seaborn') # Style To Use For Matplotlib Plots
 
Time = [i / 500 for i in range(2800)]
Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
Labels = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Training Figures (Lead II)')): # Checks If Directory Exist
    shutil.rmtree(os.path.join(Path, 'Training Figures (Lead II)')) # Removes Directory
os.mkdir(os.path.join(Path, 'Training Figures (Lead II)')) # Creates Directory

Count = 0    
for Label, Feature in tqdm.tqdm(enumerate(Training_Data), unit_scale = True, miniters = 1, desc = 'Plotting Training Data (Lead II) '): # Progress Bar
  for Sample in Feature[:100]:
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
    fig.savefig(os.path.join(Path, 'Training Figures (Lead II)', f'Figure {Count}.png'), dpi = 300)
    plt.close(fig)
 
    Count += 1


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Training Figures')): # Checks If Directory Exist
    shutil.rmtree(os.path.join(Path, 'Training Figures')) # Removes Directory
os.mkdir(os.path.join(Path, 'Training Figures')) # Creates Directory

Count = 0    
for Label, Feature in tqdm.tqdm(enumerate(Training_Data), unit_scale = True, miniters = 1, desc = 'Plotting Training Data '): # Progress Bar
  for Sample in Feature[:100]:
    df = pd.DataFrame(Sample)
 
    # Plot And Save
    fig, ax = plt.subplots(6, 2, sharex = True)
    fig.set_size_inches(37.33, 21)
    for i in range(2):
      for j in range(6):
        ax[j][i].plot(Time, df.iloc[:, i * 6 + j], 'b', label = Labels[i * 6 + j])
        ax[j][i].legend(loc = 'upper right', prop = {'size': 14})
        ax[j][i].set_ylabel('mV', fontsize = 12)  
        ax[j][i].yaxis.label.set_color('black')
        ax[j][i].tick_params(axis = 'both', colors = 'black')
      ax[j][i].set_xlabel('Time', fontsize = 12)
      ax[j][i].xaxis.label.set_color('black')
    fig.tight_layout()
    fig.suptitle(Dict[Label], fontsize = 18)
    fig.subplots_adjust(top = 0.96, right = 0.978)
    fig.savefig(os.path.join(Path, 'Training Figures', f'Figure {Count}.png'), dpi = 300)
    plt.close(fig)
 
    Count += 1


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Validation Figures (Lead II)')): # Checks If Directory Exist
    shutil.rmtree(os.path.join(Path, 'Validation Figures (Lead II)')) # Removes Directory
os.mkdir(os.path.join(Path, 'Validation Figures (Lead II)')) # Creates Directory

Count = 0    
for Label, Feature in tqdm.tqdm(enumerate(Validation_Data), unit_scale = True, miniters = 1, desc = 'Plotting Validation Data (Lead II) '): # Progress Bar
  for Sample in Feature[:100]:
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
    fig.savefig(os.path.join(Path, 'Validation Figures (Lead II)', f'Figure {Count}.png'), dpi = 300)
    plt.close(fig)
 
    Count += 1


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Validation Figures')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Validation Figures')) # Removes Directory
os.mkdir(os.path.join(Path, 'Validation Figures')) # Creates Directory
 
Count = 0
for Label, Feature in tqdm.tqdm(enumerate(Validation_Data), unit_scale = True, miniters = 1, desc = 'Plotting Validation Data '): # Progress Bar
  for Sample in Feature:
    df = pd.DataFrame(Sample)
 
    # Plot And Save
    fig, ax = plt.subplots(6, 2, sharex = True)
    fig.set_size_inches(37.33, 21)
    for i in range(2):
      for j in range(6):
        ax[j][i].plot(Time, df.iloc[:, i * 6 + j], 'b', label = Labels[i * 6 + j])
        ax[j][i].legend(loc = 'upper right', prop = {'size': 14})
        ax[j][i].set_ylabel('mV', fontsize = 12)  
        ax[j][i].yaxis.label.set_color('black')
        ax[j][i].tick_params(axis = 'both', colors = 'black')
      ax[j][i].set_xlabel('Time', fontsize = 12)
      ax[j][i].xaxis.label.set_color('black')
    fig.tight_layout()
    fig.suptitle(Dict[Label], fontsize = 18)
    fig.subplots_adjust(top = 0.96, right = 0.978)
    fig.savefig(os.path.join(Path, 'Validation Figures', f'Figure {Count}.png'), dpi = 300)
    plt.close(fig)

    Count += 1


# ## **Predictions**

# In[ ]:


Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
 
# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature[:100]]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))
 
# Take Predictions
Names = [f'Figure {index}' for index in range(x.shape[0])]
y_pred = [Dict[Label] for Label in np.argmax(model.predict(x), axis = 1)]
y_true = [Dict[Label] for Label in np.argmax(y, axis = 1)]

# Push Results To CSV File
Results = pd.DataFrame(list(zip(Names, y_true, y_pred)), columns = ['Name', 'Actual', 'Prediction'])
Results.to_csv(os.path.join(Path, 'Training Results.csv'), index = False)


# In[ ]:


Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
 
# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Validation_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))

# Take Predictions
Names = [f'Figure {index}' for index in range(x.shape[0])]
y_pred = [Dict[Label] for Label in np.argmax(model.predict(x), axis = 1)]
y_true = [Dict[Label] for Label in np.argmax(y, axis = 1)]
 
# Push Results To CSV File
Results = pd.DataFrame(list(zip(Names, y_true, y_pred)), columns = ['Name', 'Actual', 'Prediction'])
Results.to_csv(os.path.join(Path, 'Validation Results.csv'), index = False)


# ## **Save Samples Into CSV File**

# In[ ]:


if os.path.isdir(os.path.join(Path, 'Training Samples')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Training Samples')) # Removes Directory
os.mkdir(os.path.join(Path, 'Training Samples')) # Creates Directory

# Push Samples To CSV File
Count = 0    
for Label, Feature in enumerate(Training_Data):
  for Sample in Feature[:100]:
    pd.DataFrame(Sample, columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']).to_csv(os.path.join(Path, 'Training Samples', f'Figure {Count}.csv'), index = False)
    Count += 1


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Validation Samples')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Validation Samples')) # Removes Directory
os.mkdir(os.path.join(Path, 'Validation Samples')) # Creates Directory

# Push Samples To CSV File
Count = 0    
for Label, Feature in enumerate(Validation_Data):
  for Sample in Feature:
    pd.DataFrame(Sample, columns = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']).to_csv(os.path.join(Path, 'Validation Samples', f'Figure {Count}.csv'), index = False)
    Count += 1


# ## **Confusion Matrix, Accuracy, Misclassification Rate, Precision, Recall, Specificity, F1-Score**

# In[ ]:


Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}

# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature[:100]]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))

# Take Predictions
y_pred = [Dict[Label] for Label in np.argmax(model.predict(x), axis = 1)]
y_true = [Dict[Label] for Label in np.argmax(y, axis = 1)]

Matrix = confusion_matrix(y_true, y_pred)

Info_Dict = {'Accuracy': [],
             'Misclassification Rate': [],
             'Precision': [],
             'Recall': [],
             'Specificity': [],
             'F1-Score': []}

# Calculate TP, TN, FP, FN
for i in range(4):
  FN = 0
  FP = 0
  TN = 0
  TP = Matrix[i][i]
 
  for j in range(4):
    if i!=j:
      FP += Matrix[i][j]
      FN += Matrix[j][i]
 
  for j in range(4):
    if i!=j:
      for k in range(4):
        if i!=k:
          TN += Matrix[j][k]
  
  Info_Dict['Accuracy'].append((TP + TN) / (TP + TN + FP + FN))
  Info_Dict['Misclassification Rate'].append((FP + FN) / (TP + TN + FP + FN))
  Info_Dict['Precision'].append(TP / (TP + FP))
  Info_Dict['Recall'].append(TP / (TP + FN))
  Info_Dict['Specificity'].append(TN / (TN + FP))
  Info_Dict['F1-Score'].append(2 * ((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))

display(pd.DataFrame(Matrix, index = Dict.values(), columns = Dict.values()))
display(pd.DataFrame(Info_Dict, index = Dict.values()).round(4))


# In[ ]:


Dict = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'PVC'}
 
# Separates Target Variable
Data = [[Sample, index] for index, Feature in enumerate(Validation_Data) for Sample in Feature]
x, y = np.array([Sample[0] for Sample in Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Data], dtype = np.int64))

# Take Predictions
y_pred = [Dict[Label] for Label in np.argmax(model.predict(x), axis = 1)]
y_true = [Dict[Label] for Label in np.argmax(y, axis = 1)]

Matrix = confusion_matrix(y_true, y_pred)

Info_Dict = {'Accuracy': [],
             'Misclassification Rate': [],
             'Precision': [],
             'Recall': [],
             'Specificity': [],
             'F1-Score': []}
 
# Calculate TP, TN, FP, FN
for i in range(4):
  FN = 0
  FP = 0
  TN = 0
  TP = Matrix[i][i]
 
  for j in range(4):
    if i!=j:
      FP += Matrix[i][j]
      FN += Matrix[j][i]
 
  for j in range(4):
    if i!=j:
      for k in range(4):
        if i!=k:
          TN += Matrix[j][k]
  
  Info_Dict['Accuracy'].append((TP + TN) / (TP + TN + FP + FN))
  Info_Dict['Misclassification Rate'].append((FP + FN) / (TP + TN + FP + FN))
  Info_Dict['Precision'].append(TP / (TP + FP))
  Info_Dict['Recall'].append(TP / (TP + FN))
  Info_Dict['Specificity'].append(TN / (TN + FP))
  Info_Dict['F1-Score'].append(2 * ((TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN))))

display(pd.DataFrame(Matrix, index = Dict.values(), columns = Dict.values()))
display(pd.DataFrame(Info_Dict, index = Dict.values()).round(4))

