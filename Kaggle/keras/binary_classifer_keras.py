
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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[4]:

# Read data file
data = pd.read_csv("Loan_Training.csv", header=0)
seed = 5
numpy.random.seed(seed)


# In[5]:

# Take a look at the data
print(data.head(1))


# In[6]:

# Take a look at the types of data
data.info()


# In[7]:

# Column Unnamed : 32 holds only null values, so it is of no use to us. We simply drop that column.
#data.drop("Unnamed: 32",axis=1,inplace=True)
#data.drop("id", axis=1, inplace=True)


# In[8]:

# Check whether the column has been dropped
data.columns


# In[9]:

# Select the columns to use for prediction in the neural network
prediction_var = ['Loan ID', 'Amount Requested', 'Amount Funded By Investors',
       'Interest Rate', 'Loan Length', 'CREDIT Grade', 'Loan Title',
       'Loan Purpose', 'Monthly PAYMENT', 'Total Amount Funded',
       'Debt-To-Income Ratio', 'City', 'State', 'Home Ownership',
       'Monthly Income', 'FICO Range', 'Earliest CREDIT Line',
       'Open CREDIT Lines', 'Total CREDIT Lines', 'Revolving CREDIT Balance',
       'Revolving Line Utilization', 'Inquiries in the Last 6 Months',
       'Accounts Now Delinquent', 'Delinquent Amount',
       'Delinquencies (Last 2 yrs)', 'Months Since Last Delinquency',
       'Public Records On File', 'Months Since Last Record', 'Education',
       'Employment Length']
X = data[prediction_var].values
Y = data.Status.values


# In[10]:

# Diagnosis values are strings. Changing them into numerical values using LabelEncoder.
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[11]:

# Baseline model for the neural network. We choose a hidden layer of 10 neurons. The lesser number of neurons helps to eliminate the redundancies in the data and select the more important features.
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:

# Evaluate model using standardized dataset.
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=3, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:
