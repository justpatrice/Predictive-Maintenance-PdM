
# coding: utf-8

# # Building Predictive Maintenance Model

# - London South Bank University, Januar 19, 2019
# - School of Engineering, Data Science MSc
# - Module: Machine Learning
# - Lecturer: Manik Gupta & Oswaldo Cadena
# - Author: Patric Oliver Weber
# - Version: 1.2
# 
# # 1.0 Introduction
# With the given knowledge and tools of trade received from lectures by LSBU about predictive-modelling, I am going to gain more in-depth understanding of the Python packages and the Jupyter Notebook by conducting the exploratory data analysis and modeling  by a self selected  Dataset to Predict Maintenance and will apply machine learning techniques and gain valuable insights into data processing. 
# 
# This experiment demonstrates the steps in building a predictive maintenance solution. The data had been taken because the Author was working on a Predictive Maintenance Project before attending LSBU. At this time we were using Azure Machine Learning Studio and Machine Logs from TrueBeam machines to predict failures *(see Appendix for Manamement Summary)*. However, with the data we got, it was not possible to generate a working model, and most time were spent on creating the dataset. Due to the fact, I know this NASA dataset, because I have used it as a guideline and template,  I decided to use it and transfer the programming tasks into Python. The research of the dataset is roughly guided through the Cross-industry standard process for data mining,  known as CRISP-DM.
# In this Jupyter Notebook, I would like to show my transferrable skills in Machine Learning and how to get along with different libraries. Further, I will summarise the findings of this data set and suggest further course of action. 
# 
# Agenda: 
# - Introduction
# - Data Understanding
# - Data Preprocessing
# - Data Modelling
# - Data Evaluation
# - Conclustion
# - Bibliography
# - Appendix
# 
# 
# 

# ### Conda Environment *(conda list)*
# 

# - python                    3.6.8
# - numpy                     1.15.4
# - pandas                    0.23.4
# - pip                       18.1
# - scipy                     1.1.0
# - seaborn                   0.9.0
# - tensorflow                1.12.0
# - matplotlib                3.0.2
# - scikit-learn              0.20.2
# - Keras                     2.2.4
# - h5py                      2.9.0

# ### Conda Environment initialisation *(Anaconda Prompt)*

# **Optional Tasks prior to run the dataset.**
# 
# - python -m pip install --upgrade pip
# - pip install keras
# - pip install --upgrade tensorflow
# - conda update --all
# - pip install wget

#  
# 
# 
# # Data Understanding - Turbofan Engine Degradiation Simulation Data Set
# This predictive maintenance template focuses on the techniques used to predict when an in-service machine will fail, so that maintenance can be planned in advance. The Dataset comes from the the NASA [1]. 
# 
# **Experiment a binary classification task**
# 
# Three modeling solutions are known to do with in this template to accomplish the following tasks [2]: 
# 
# - Regression: Predict the Remaining Useful Life (RUL), or Time to Failure (TTF).
# - Binary classification: Predict if an asset will fail within certain time frame (e.g. days).
# - Multi-class classification: Predict if an asset will fail in different time windows: E.g., fails in window [1, w0] days; fails in the window [w0+1,w1] days; not fail within w1 days
# 
# The time units mentioned above can be replaced by working hours, cycles, mileage, transactions, etc. based on the actual scenario.
# 
# **However, In this Notebook there is only the binary classification shown, which uses a LTSM framework.**
# 
# - [1] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-
# - [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models 
# https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2

# The Dataset is already pre splited into Train and Testing data
# 
# - Training data: It is the aircraft engine run-to-failure data.
# - Testing data: It is the aircraft engine operating data without failure events recorded.
# - Ground truth data: It contains the information of true remaining cycles for each engine in the testing data.

# In[1]:


# The Algorithm 
# start conda environment console and tipe in pip install keras and pip install --upgrade tensorflow
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

get_ipython().run_line_magic('matplotlib', 'inline # Im dokumnet rein.')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


# Setting seed for reproducability
np.random.seed(1234)  
PYTHONHASHSEED = 0
# With this statement I would like to make sure that you get the same result as I do. 


# ### Data Ingestion *(Optional)*

# In this section I will show how we can download the data within the python environment. 

# In[4]:


# Data Ingestion - reading the datasets from Azure blob
import wget
url = 'http://azuremlsamples.azureml.net/templatedata/PM_train.txt'
filename = wget.download(url)
url = 'http://azuremlsamples.azureml.net/templatedata/PM_test.txt'
filename = wget.download(url)
url = 'http://azuremlsamples.azureml.net/templatedata/PM_truth.txt'
filename = wget.download(url)
# The files are now stored in the working directory.


# ## Load the Datasets

# ### Read the training data (df_train)

# In[5]:


# read training data - it is the aircraft engine run-to-failure
df_train = pd.read_csv('PM_train.txt', sep=" ", header=None)
# Drop the two last columns with "NaN"
df_train.drop(df_train.columns[[26, 27]], axis=1, inplace=True)
# add columns
df_train.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

#Show the data
df_train.tail()


# In[6]:


# Sort the data to better see the machine id and the cycles.
df_train = df_train.sort_values(['id','cycle'])
df_train.head()


# ### Read the test data (df_test)

# In[7]:


# read test data - It is the aircraft engine operating data without failure events recorded 
df_test = pd.read_csv('PM_test.txt', sep=" ", header=None)
df_test.drop(df_test.columns[[26, 27]], axis=1, inplace=True)
df_test.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                    's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                    's15', 's16', 's17', 's18', 's19', 's20', 's21']
df_test.tail()


# ### Read the truth data (df_truth)

# In[8]:


# read ground truth data - This containes the information of the true remaining cycles for each in the testing data.
df_truth = pd.read_csv('PM_truth.txt', sep=" ", header=None)
df_truth.drop(df_truth.columns[[1]], axis=1, inplace=True)
df_truth.tail()


# # Data preprocessing

# Our first step is to generate the lables for the training data. This is Remaining Useful Life (RUL). This will say how much cycles the device was having until it failed. In a nutshell it is the just the cycle but reversed. Here we will for both the following:
# - Join Data
# - Generate Label Column
# - Normalize

# ### Join Data

# In[9]:


# Data Labeling - we will generate column RUL(Remaining Usefull Life of Time to Failure)
rul = pd.DataFrame(df_train.groupby('id')['cycle'].max()).reset_index()
# rename the cylce to see perfctly what was the max of each machine ;-)
rul.columns = ['id', 'max']
rul.head()


# In[10]:


# Merge (Left inner Join) the RUL together with the train set. 
df_train = df_train.merge(rul, on=['id'], how='left')
# Calculate on the max the cycle for RUL
df_train['RUL'] = df_train['max'] - df_train['cycle']
# Drop the max
df_train.drop('max', axis=1, inplace=True)
df_train.head()


# This has shown a nice example how to merge the RUL. There are many ways how this could be implementet. 

# ### Generate Label colums for training data

# In this section we will generate the Label for the training data so that we can make in the end a binary classifier.
# 
# We are only doing the label1, this is for the binary classification.
# So, we try to ask the following question: is a specific engine going to fail within w1 cycle?

# In[11]:


# Let's generate the Label for training data

w1 = 30 # this means if machine will fail within 30 cycles
w0 = 15 # this means if machine will fail within 15 cycles

df_train['label1'] = np.where(df_train['RUL'] <= w1, 1, 0)
# this is easy like the following statement. However I also have seen else.
# df_train['label2'] = np.where(df_train['RUL'] <= w0, 1 ,0)
df_train['label2'] = df_train['label1']
df_train.loc[df_train['RUL'] <= w0, 'label2'] = 2
# so this counts on behlaf of label one and points out with the statement 2 that it is more critical. 
df_train.tail()


# In this predictive maintenance example, the cycle column will aslo used for the training and fitting. Therefore we will include the cycle column. 

# ### Normalize data with min-max

# In[12]:


from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# MinMax normalization
df_train['cycle_norm'] = df_train['cycle']
cols_normalize = df_train.columns.difference(['id','cycle','RUL','label1','label2'])
min_max_scaler = preprocessing.MinMaxScaler()
df_train_norm = pd.DataFrame(min_max_scaler.fit_transform(df_train[cols_normalize]), 
                             columns=cols_normalize, 
                             index=df_train.index)
join_df = df_train[df_train.columns.difference(cols_normalize)].join(df_train_norm)
df_train = join_df.reindex(columns = df_train.columns)
df_train.head()


# As a next step we will prepare the test data. Hence, we also have to normalize the test data by using the well known and famous MinMax normalization algorithm. This should really be a nobrainer and only the variables needs to be changed. Having said that let's get back to work.

# ### Perpare the test data

# In[13]:


# Normalize the Test data. 
df_test['cycle_norm'] = df_test['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(df_test[cols_normalize]), 
                            columns=cols_normalize, 
                            index=df_test.index)
test_join_df = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
df_test = test_join_df.reindex(columns = df_test.columns)
df_test = df_test.reset_index(drop=True)
df_test.head()


# In[14]:


# generate max column for test data
rul = pd.DataFrame(df_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
df_truth.columns = ['more']
df_truth['id'] = df_truth.index + 1
df_truth['max'] = rul['max'] + df_truth['more']
df_truth.drop('more', axis=1, inplace=True)


# In[15]:


# generate the RUL for the test data 
df_test = df_test.merge(df_truth, on=['id'], how='left')
df_test['RUL'] = df_test['max'] - df_test['cycle']
df_test.drop('max', axis=1, inplace=True)


# In[16]:


# generate the labels for the test data as well -.-
df_test['label1'] = np.where(df_test['RUL'] <= w1, 1, 0)
df_test['label2'] = df_test['label1']
df_test.loc[df_test['RUL'] <= w0, 'label2'] = 2
df_test.tail()


# 
# In the rest of the notebook, we train an LSTM network that we will compare to the results in Predictive Maintenance Template Step 2B of 3 where a series of machine learning models are used to train and evaluate the binary classification model that uses column "label1" as the label.

# # Modelling

# In this example of my dataset I will use an advanced algorithm which has not been taught at LSBU. 
# It is a neural network , with also a focus on the sime-series domain.
# 
# Traditional predictive maintenance machine learning models are most of the time based on proper feature engineering. As the data comes from the machine engineers may need to know you for what each sensor stands. To distinguish the right features, domain expertise needs to be used. Especially this makes it hard to reuse such a template in Pdm. However, one positive thing is what makes it still attractive is applying deep learning in the predictive maintenance project. These learners are a holy grail because they are automatically able to extract the right features form the data, this eliminates or reduces the need for manual feature engineering.
# 
# But! With my expertise from the Silicon Valley I have seen that it needs engineers because in the mess of the data it was even hard to distinguish the labels.
# 
# ### LTSM explained
# When using LSTMs in the time-series domain, one important parameter to pick is the sequence length which is the window for LSTMs to look back. 
# This may be viewed as similar to picking window_size = 5 cycles for calculating the rolling features in the Predictive Maintenance Template which are rolling mean and rolling standard deviation for 21 sensor values. The idea of using LSTMs is to let the model extract general features out of the sequence of sensor values in the window rather than engineering those manually. The expectation is that if there is a pattern in these sensor values within the window before failure, the pattern should be encoded by the LSTM.

# In[17]:


# pick a large window size of 50 cycles
sequence_length = 50


# Let's first look at an example of the sensor values 50 cycles prior to the failure for engine id 3. We will be feeding LSTM network this type of data for each time step for each engine id.

# In[18]:


# preparing data for visualizations 
# window of 50 cycles prior to a failure point for engine id 3

# select the machine
engine_id3 = df_test[df_test['id'] == 3]
engine_id3_50cycleWindow = engine_id3[engine_id3['RUL'] <= engine_id3['RUL'].min() + 50]
cols1 = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
engine_id3_50cycleWindow1 = engine_id3_50cycleWindow[cols1]
cols2 = ['s11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
engine_id3_50cycleWindow2 = engine_id3_50cycleWindow[cols2]


# In[19]:


# plotting sensor data for engine ID 3 prior to a failure point - sensors 1-10 
ax1 = engine_id3_50cycleWindow1.plot(subplots=True, sharex=True, figsize=(20,20))


# In[20]:


# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax2 = engine_id3_50cycleWindow2.plot(subplots=True, sharex=True, figsize=(20,20))


# Keras LSTM layers expect an input in the shape of a numpy array of 3 dimensions (samples, time steps, features) where samples is the number of training sequences, time steps is the look back window or sequence length and features is the number of features of each sequence at each time step.
# 
# 

# In[21]:


# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]


# In[22]:


# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)


# In[23]:


# generator for the sequences
seq_gen = (list(gen_sequence(df_train[df_train['id']==id], sequence_length, sequence_cols)) 
           for id in df_train['id'].unique())


# In[24]:


# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
seq_array.shape


# In[25]:


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# In[26]:


# generate labels
label_gen = [gen_labels(df_train[df_train['id']==id], sequence_length, ['label1']) 
             for id in df_train['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape


# ## LSTM Network (Long Short-Term Memory)
# 

# Next, we build a deep network. The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. Dropout is also applied after each LSTM layer to control overfitting. Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem. :) 

# In[27]:


# build the network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[28]:


print(model.summary())


# In[29]:


get_ipython().run_cell_magic('time', '', "history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,\n          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')])")


# # Evaluation 

# ### Training Set

# In[30]:


# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))


# In[31]:


# make predictions and compute confusion matrix
y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)
y_true = label_array
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true, y_pred)
cm


# In[32]:


# compute precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)


# ### Test Set

# 
# Next, we look at the performance on the test data. In the Predictive Maintenance Template Step 1 of 3, only the last cycle data for each engine id in the test data is kept for testing purposes. In order to compare the results to the template, we pick the last sequence for each id in the test data.

# In[33]:


seq_array_test_last = [df_test[df_test['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in df_test['id'].unique() if len(df_test[df_test['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
seq_array_test_last.shape

#print(label_array_test_last.shape)
#print("label_array_test_last")
#print(label_array_test_last)


# In[34]:


# it is used to take only the labels of the sequences that are at least 50 long
y_mask = [len(df_test[df_test['id']==id]) >= sequence_length for id in df_test['id'].unique()]


# In[35]:


label_array_test_last = df_test.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
label_array_test_last.shape


# In[36]:


print(seq_array_test_last.shape)
print(label_array_test_last.shape)


# In[37]:


# test metrics
scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
print('Accurracy: {}'.format(scores_test[1]))


# In[38]:


# make predictions and compute confusion matrix
y_pred_test = model.predict_classes(seq_array_test_last)
y_true_test = label_array_test_last
print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true_test, y_pred_test)
cm


# In[39]:


# compute precision and recall
precision_test = precision_score(y_true_test, y_pred_test)
recall_test = recall_score(y_true_test, y_pred_test)
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )


# In[40]:


# compute AUC (area under the curve)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred)
print( 'auc = ', auc)


# In[41]:


results_df = pd.DataFrame([[scores_test[1],precision_test,recall_test,f1_test],
                          [0.94, 0.952381, 0.8, 0.869565]],
                         columns = ['Accuracy', 'Precision', 'Recall', 'F1-score'],
                         index = ['LSTM',
                                 'Template Best Model'])
results_df


# In[42]:


# summarize history for Accuracy
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_accuracy.png")


# In[43]:


# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_loss.png")


# In[44]:


# Plot in blue color the predicted data and in green color the
# actual data to verify visually the accuracy of the model.
fig_verify = plt.figure(figsize=(10, 5))
plt.plot(y_pred_test, color="blue")
plt.plot(y_true_test, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()
fig_verify.savefig("model_verify.png")


# # Conclusion

# My tutorial has covered the basics of using deep learning in a particular field. I was first thinking of applying Random Forest, but by consulting many research pages, I decided to improve my knowledge in the deep learning field. It was great that I have found out that many predictive maintenance problems usually involve a variety of data sources that need to be taken into account in this domain. Further, it is vital to tune the models for the right parameters and windows. 
# 
# With the template provided form Microsoft, I was able to transfer my skills into Python and also made good use of consulting the documentation of Keras and looking at code from other Data Scientists on GitHub.
# 
# However, the data set is still sometimes wholly new to me, and in some parts, I was struggling to understand the entire data. I wanted to plot some graphs but was unable to create the right ones, which shows lack of essential Pandas training and first dataset observations. With more time I would try hard to understand the data perfectly, as well as trying out some different window sizes, a different deep learning architecture with different layers and nodes and tune hyperparameters of the network and increase the obvervations. Nevertheless, this all sounds still Utopia to me; I might start first with a Predictive Maintenance Modelling guide from scratch. 
# 
# Lastly, I would like to mention that the dataset is perfect, but still actually addresses the wrong Business goals.
# The aim of predictive maintenance, shorten as (PdM) is first to predict when equipment failure might occur (Regression), secondly then prevent the incident of failure by performing maintenance when PdM is working effectively as a maintenance strategy; maintenance is only performed on machines when it is required. To summarize, with PdM you should figure out wich sensor has failed and work together with engnineers to improve this component, instead of doing Ad-hoc pred Maintenance. 

# # Bibliogrpahy

# - cmdline. (2018, June 15). How To Split A Column or Column Names in Pandas and Get Part of it? Retrieved January 7, 2019, from http://cmdlinetips.com/2018/06/how-to-split-a-column-or-column-names-in-pandas-and-get-part-of-it/
# 
# - Contribute to Azure/Azure-MachineLearning-ClientLibrary-Python development by creating an account on GitHub. (2018). - Python, Microsoft Azure. Retrieved from https://github.com/Azure/Azure-MachineLearning-ClientLibrary-Python (Original work published 2015)
# 
# - CSV to dataframe conversion using python. (n.d.). Retrieved January 7, 2019, from https://social.msdn.microsoft.com/Forums/en-US/afccb6b2-34a1-447b-b45e-f16373818402/csv-to-dataframe-conversion-using-python?forum=MachineLearning
# 
# - Google Colaboratory. (n.d.-a). Retrieved January 11, 2019, from https://colab.research.google.com/drive/1tjIOud2Cc6smmvZsbl-QDBA6TLA2iEtd#scrollTo=j1T8y5qA91x8
# 
# - Griffo, U. (2019a). Example of Multiple Multivariate Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras.: umbertogriffo/Predictive-Maintenance-using-LSTM. Python. Retrieved from https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM (Original work published 2017)
# 
# - Intelligent Systems Division. (n.d.). Retrieved January 7, 2019, from https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan
# 
# - LSTMS for Predictive Maintenance. Contribute to Azure/lstms_for_predictive_maintenance development by creating an account on GitHub. (2019). Jupyter Notebook, Microsoft Azure. Retrieved from https://github.com/Azure/lstms_for_predictive_maintenance (Original work published 2017)
# 
# - Open source documentation of Microsoft Azure. Contribute to MicrosoftDocs/azure-docs development by creating an account on GitHub. (2019). PowerShell, Microsoft Docs. Retrieved from https://github.com/MicrosoftDocs/azure-docs (Original work published 2016)
# 
# - Ph.D, J. X. (2018). In this repo we will show you how to set up Databricks clsuter and run Python 3 with keras and tensorflow for LSTM model against the same dataset for the predictive maintenance. : JackXueIndiana/D.. Jupyter Notebook. Retrieved from https://github.com/JackXueIndiana/Databricks-TensorFlow-LSTM-Predictive-Maintenance (Original work published 2018)
# 
# - Predictive Maintenance aircraft data. (n.d.). Retrieved January 11, 2019, from https://kaggle.com/maternusherold/pred-maintanance-data
# 
# - Predictive Maintenance and Analytics Solutions. (n.d.). Retrieved January 18, 2019, from //www.schneider-electric.co.in/en/work/services/field-services/predictive/index.jsp
# 
# - Predictive Maintenance Modelling Guide R Notebook. (n.d.). Retrieved January 18, 2019, from https://gallery.azure.ai/Notebook/Predictive-Maintenance-Modelling-Guide-R-Notebook-1
# 
# - Predictive Maintenance: Step 2A of 3, train and evaluate regression models. (n.d.). Retrieved January 7, 2019, from https://gallery.azure.ai/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
# 
# - Predictive Maintenance Template. (n.d.). Retrieved January 7, 2019, from https://gallery.azure.ai/Collection/Predictive-Maintenance-Template-3
# 
# - python - How to load a tsv file into a Pandas DataFrame? (n.d.). Retrieved January 7, 2019, from https://stackoverflow.com/questions/9652832/how-to-load-a-tsv-file-into-a-pandas-dataframe
# 
# - Zhang, Y. (n.d.). Building Predictive Maintenance Solutions with Azure Machine Learning. Juli 2015. Retrieved from https://azure.microsoft.com/en-gb/resources/videos/building-predictive-maintenance-solutions-with-azure-machine-learning/

# # Appendix:

# ## Managment Summary - Varian Machine Learning 6 Week Silicon Valley with 4 Fellow-Students

# Confidential 
