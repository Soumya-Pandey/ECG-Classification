#!/usr/bin/env python
# coding: utf-8

# ## **Load Libraries**

# In[ ]:


get_ipython().system(u'pip install zipfile38')
 
import os
import tqdm
import math
import random
import pickle
import shutil
import numpy as np
import pandas as pd
import zipfile38 as zipfile
 
from scipy import io
from urllib import request
from tensorflow.keras.utils import to_categorical


# ## **Mount Drive**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive') # Mounts Drive In Content Directory (Under Files Section)


# In[ ]:


Path = r'/content/drive/MyDrive/ECG ML'


# ## **Download Dataset**

# In[ ]:


# Progress Bar For Downloading
def my_hook(t): 
    '''Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook = reporthook)
    '''
    last_b = [0]
 
    def update_to(b = 1, bsize = 1, tsize = None):
        '''
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        '''
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed
 
    return update_to


# In[ ]:


if os.path.isdir(os.path.join(Path, 'Dataset')): # Checks If Directory Exist
  shutil.rmtree(os.path.join(Path, 'Dataset')) # Removes Directory
os.mkdir(os.path.join(Path, 'Dataset')) # Creates Directory

# Downloads Dataset
for index in range(3):
  with tqdm.tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = f'Downloading TrainingSet{index + 1}.zip ') as t: # Progress Bar
    request.urlretrieve(f'http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet{index + 1}.zip', os.path.join(Path, 'Dataset', f'TrainingSet{index + 1}.zip'), my_hook(t)) # Connects And Downloads Data


# ## **Unpack Dataset**

# In[ ]:


# Unzip Archive Files
for Archive in ['TrainingSet1.zip', 'TrainingSet2.zip', 'TrainingSet3.zip']:
  with zipfile.ZipFile(os.path.join(Path, 'Dataset', Archive)) as zf:
    for member in tqdm.tqdm(zf.infolist(), unit_scale = True, miniters = 1, desc = f'Unpacking {Archive} '): # Progress Bar
      zf.extract(member, os.path.join(Path, 'Dataset'))


# ## **Convert and Merge Dataset**

# In[ ]:


os.mkdir(os.path.join(Path, 'Dataset', 'Converted')) # Create Directory
 
for TrainingSet in ['TrainingSet1', 'TrainingSet2', 'TrainingSet3']:
  for File_Name in tqdm.tqdm(os.listdir(os.path.join(Path, 'Dataset', TrainingSet)), unit_scale = True, miniters = 1, desc = f'Converting {TrainingSet} '):
    if File_Name[0] == 'A':
      pd.DataFrame(io.loadmat(os.path.join(Path, 'Dataset', TrainingSet, File_Name))['ECG'][0][0][2].T, columns = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']).to_csv(os.path.join(Path, 'Dataset', 'Converted', f'{os.path.splitext(File_Name)[0]}.csv'), index = False) # Reads MATLAB Files And Converts Them To CSV File
shutil.copyfile(os.path.join(Path, 'Dataset', 'TrainingSet3', 'REFERENCE.csv'), os.path.join(Path, 'Dataset', 'Converted', 'Reference.csv')) # Copy File (Reference.csv)


# ## **Select Required Features and Split Dataset**

# In[ ]:


# Features To Include In Model
Feature_List = [1, 4, 5, 7]
Feature_List.sort()
 
Dict = {}
for index, Feature in enumerate(Feature_List):
  Dict[Feature] = index


# In[ ]:


# Reads Reference.csv File
df = pd.read_csv(os.path.join(Path, 'Dataset', 'Converted', 'Reference.csv'), index_col = 'Recording').sort_index()

# Selects Required Features
First_label = df[df['First_label'].isin(Feature_List)].drop(columns = ['Second_label', 'Third_label'])
First_label.columns = ['Label']
 
Second_label = df[df['Second_label'].isin(Feature_List)].drop(columns = ['First_label', 'Third_label'])
Second_label = Second_label[np.logical_not(Second_label.index.isin(First_label.index))]
Second_label.columns = ['Label']
 
Third_label = df[df['Third_label'].isin(Feature_List)].drop(columns = ['First_label', 'Second_label'])
Third_label = Third_label[np.logical_not(Third_label.index.isin(pd.Series(np.concatenate([First_label.index, Second_label.index]))))]
Third_label.columns = ['Label']
 
# Sort Dataset According To Number Of Samples
df = pd.concat([First_label, Second_label, Third_label], axis = 0).convert_dtypes().sort_index()
Length_Dict = {File_Name: np.genfromtxt(os.path.join(Path, 'Dataset', 'Converted', f'{File_Name}.csv'), dtype = np.float64, delimiter = ',', skip_header = 1).shape[0] for File_Name in tqdm.tqdm(df.index, unit_scale = True, miniters = 1, desc = 'Calculating Lengths ')}

df = df.reindex(sorted(df.index, key = lambda File_Name: Length_Dict[File_Name]))
Max = pd.read_csv(os.path.join(Path, 'Dataset', 'Converted', f'{df.index[-1]}.csv')).shape[0] # Gets The Maximum Number Of Samples Available In Any Single File


# In[ ]:


# Separate Validation Files
Count = [0] * len(Feature_List)
Validation_Files = [[] for _ in range(len(Feature_List))]

index = 0
while sum(Count) != 100:
  File_Name = df.index[index]
  Label = df.loc[File_Name, 'Label']
  
  if Count[Dict[Label]] < 25:
    Validation_Files[Dict[Label]].append(File_Name)
    Count[Dict[Label]] += 1
 
  index += 1
 
Validation_Files = [File_Name for Feature in Validation_Files for File_Name in Feature]


# In[ ]:


# Separate Dataframe
train_df = df[np.logical_not(df.index.isin(Validation_Files))]
validation_df = df[df.index.isin(Validation_Files)]


# ## **Augment Dataset**

# In[ ]:


# Augment Dataset Using Fixed Overlapping Window

Max = int(Max / 1400) - 1
Augmented_Data = [[] for _ in range(len(Feature_List))]
Count = [[] for _ in range(len(Feature_List))]

for File_Name in tqdm.tqdm(train_df.index, unit_scale = True, miniters = 1, desc = 'Loading Training Data '): # Progress Bar
  File = np.genfromtxt(os.path.join(Path, 'Dataset', 'Converted', f'{File_Name}.csv'), dtype = np.float64, delimiter = ',', skip_header = 1) # Read Files
  Label = train_df.loc[File_Name, 'Label']
  Length = int(File.shape[0] / 1400) - 1
 
  # Augment
  File_Data = [File[index * 1400:(index + 2) * 1400] for index in range(0, Length, 2)]
  File_Data.extend([np.empty([0]) for _ in range(int(Max / 2) - math.ceil(Length / 2))])
  File_Data.extend([File[index * 1400:(index + 2) * 1400] for index in range(1, Length, 2)])
  File_Data.extend([np.empty([0]) for _ in range(int(Max / 2) - math.floor(Length / 2))])
  
  Count[Dict[Label]].append(math.ceil(Length / 2) + math.floor(Length / 2))  
  Augmented_Data[Dict[Label]].append(File_Data)


# In[ ]:


# Load Validation Data

Validation_Data = [[] for _ in range(len(Feature_List))]
 
for File_Name in tqdm.tqdm(validation_df.index, unit_scale = True, miniters = 1, desc = 'Loading Validation Data '): # Progress Bar
  File = np.genfromtxt(os.path.join(Path, 'Dataset', 'Converted', f'{File_Name}.csv'), dtype = np.float64, delimiter = ',', skip_header = 1)
  Label = validation_df.loc[File_Name, 'Label']
  
  # Append Directly
  Validation_Data[Dict[Label]].append(File[0:2800])


# ## **Balance Dataset**

# In[ ]:


# This Balances Dataset (Equal Number Of Samples Of Each Class)

Feature_List = [index for index in range(len(Augmented_Data))]
Max = min(sum(Feature) for Feature in Count) # Min Number Of Samples Present In Any Class

Count = [0] * len(Feature_List)
LENGTHS = [len(Feature) for Feature in Augmented_Data]
Training_Data = [[] for _ in range(len(Feature_List))]

# Main Loop
index = 0
Length = len(Feature_List)
while Length:
  Feature = Augmented_Data[Feature_List[index]]
 
  INDEX = 0
  LENGTH = LENGTHS[Feature_List[index]]
  while INDEX < LENGTH:
    File = Feature[INDEX]
    
    if File[0].shape[0] != 0:
      Training_Data[Feature_List[index]].append(File[0])  
      Count[Feature_List[index]] += 1
      if Count[Feature_List[index]] == Max:
        break
    
    del File[0]     
    if not len(File):
      del Feature[INDEX]
      INDEX -= 1
      LENGTH -= 1

    INDEX += 1
 
  LENGTHS[Feature_List[index]] = LENGTH
 
  if Count[Feature_List[index]] == Max:
    del Feature_List[index]
    index -= 1
    Length -= 1
 
  index += 1
 
  if index == Length:
    index = 0


# ## **Save Processed Dataset**

# In[ ]:


# Save Data List In Pickle Format
pickle_out = open(os.path.join(Path, 'Dataset', 'Training_Data.pickle'), 'wb')
pickle.dump(Training_Data, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(Path, 'Dataset', 'Validation_Data.pickle'), 'wb')
pickle.dump(Validation_Data, pickle_out)
pickle_out.close()


# ## **Shuffle Dataset**

# In[ ]:


# Restructure Data List
Training_Data = [[Sample, index] for index, Feature in enumerate(Training_Data) for Sample in Feature]

# Shuffle Data
random.shuffle(Training_Data)


# In[ ]:


# Restructure Data List
Validation_Data = [[Sample, index] for index, Feature in enumerate(Validation_Data) for Sample in Feature]

# Shuffle Data
random.shuffle(Validation_Data)


# ## **Save Shuffled Dataset**

# In[ ]:


# Separate Target Variable
x_train, y_train = np.array([Sample[0] for Sample in Training_Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Training_Data], dtype = np.int64))
x_validation, y_validation = np.array([Sample[0] for Sample in Validation_Data], dtype = np.float64), to_categorical(np.array([Sample[1] for Sample in Validation_Data], dtype = np.int64))


# In[ ]:


# Save Data List In Pickle Format
pickle_out = open(os.path.join(Path, 'Dataset', 'x_train.pickle'), 'wb')
pickle.dump(x_train, pickle_out)
pickle_out.close()
 
pickle_out = open(os.path.join(Path, 'Dataset', 'y_train.pickle'), 'wb')
pickle.dump(y_train, pickle_out)
pickle_out.close()


# In[ ]:


# Save Data List In Pickle Format
pickle_out = open(os.path.join(Path, 'Dataset', 'x_validation.pickle'), 'wb')
pickle.dump(x_validation, pickle_out)
pickle_out.close()
 
pickle_out = open(os.path.join(Path, 'Dataset', 'y_validation.pickle'), 'wb')
pickle.dump(y_validation, pickle_out)
pickle_out.close()

