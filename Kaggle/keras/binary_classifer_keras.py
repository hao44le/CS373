
# coding: utf-8

# In[1]:

# Import libraries for data wrangling, preprocessing and visualization
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:

# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[3]:

# Read data file
data = pd.read_csv("Loan_Training.csv", header=0)
seed = 5
numpy.random.seed(seed)


# In[4]:

# Take a look at the data
print(data.head(1))


# In[5]:

# Take a look at the types of data
data.info()


# In[6]:

# Column Unnamed : 32 holds only null values, so it is of no use to us. We simply drop that column.
data.drop("Loan Title",axis=1,inplace=True)
data.drop("Loan ID", axis=1, inplace=True)
data.drop("Education",axis=1,inplace=True)
data.drop("Earliest CREDIT Line",axis=1,inplace=True)
data.drop("City",axis=1,inplace=True)
data.drop("State",axis=1,inplace=True)

# In[7]:

# Check whether the column has been dropped
data.columns


# In[8]:

# Select the columns to use for prediction in the neural network
prediction_var = ['Amount Requested', 'Amount Funded By Investors', 'Interest Rate',
       'Loan Length', 'CREDIT Grade', 'Loan Purpose', 'Monthly PAYMENT',
       'Total Amount Funded', 'Debt-To-Income Ratio',
       'Home Ownership', 'Monthly Income', 'FICO Range', 'OpenCREDITLines', 'Total CREDIT Lines',
       'Revolving CREDIT Balance', 'Revolving Line Utilization',
       'Inquiries in the Last 6 Months', 'Accounts Now Delinquent',
       'Delinquent Amount', 'Delinquencies (Last 2 yrs)',
       'Months Since Last Delinquency', 'Public Records On File',
       'Months Since Last Record', 'Employment Length']
X = data[prediction_var]
Y = data.Status.values


# In[9]:

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[10]:

# Baseline model for the neural network. We choose a hidden layer of 10 neurons. The lesser number of neurons helps to eliminate the redundancies in the data and select the more important features.
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=len(prediction_var), kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# In[11]:

from column_encoders import MultiColumnLabelEncoder


# In[12]:

# X = X.dropna()
pd.options.mode.chained_assignment = None  # default='warn'
dropped_X = X
dropped_X["OpenCREDITLines"] = dropped_X["OpenCREDITLines"].fillna(0)
dropped_X["Total CREDIT Lines"] = dropped_X["Total CREDIT Lines"].fillna(0)
dropped_X["Revolving CREDIT Balance"] = dropped_X["Revolving CREDIT Balance"].fillna(0)
dropped_X["Inquiries in the Last 6 Months"] = dropped_X["Inquiries in the Last 6 Months"].fillna(0)
dropped_X["Accounts Now Delinquent"] = dropped_X["Accounts Now Delinquent"].fillna(0)
dropped_X["Delinquent Amount"] = dropped_X["Delinquent Amount"].fillna(0)
dropped_X["Delinquencies (Last 2 yrs)"] = dropped_X["Delinquencies (Last 2 yrs)"].fillna(0)
dropped_X["Months Since Last Delinquency"] = dropped_X["Months Since Last Delinquency"].fillna(0)
dropped_X["Public Records On File"] = dropped_X["Public Records On File"].fillna(0)
dropped_X["Months Since Last Record"] = dropped_X["Months Since Last Record"].fillna(0)
# dropped_X["Months Since Last Record"] = dropped_X["Months Since Last Record"].fillna(0)


# In[13]:

X = MultiColumnLabelEncoder(columns = prediction_var).fit_transform(dropped_X)


# In[16]:

encoded_Y
encoded_Y.shape


# In[17]:

# Evaluate model using standardized dataset.
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=3, shuffle=True)
print(kfold)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:
