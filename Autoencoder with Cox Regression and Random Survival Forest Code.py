# -*- coding: utf-8 -*-
"""

@author: Eng. Mostafa Samy Atlam

"""

import time
start_time = time.time()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# Importing the libraries
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dropout
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))

"""
The features are seperated to three csv files. After reading the three files they are added
to each other using concat command.
"""

#Loading Data.
one = pd.read_csv('one.csv')
two = pd.read_csv('two.csv')
three = pd.read_csv('three.csv')
event = pd.read_csv('events.csv')

#Fill NAN.
one = one.fillna(one.mean())
two = two.fillna(two.mean())
three = three.fillna(three.mean())

#Combine the smaller datasets, forming a dataframe of all features and events for all samples .
df = pd.concat([one, two], axis=1, join='inner')
df = pd.concat([df, three], axis=1, join='inner')
df = pd.concat([df, event], axis=1, join='inner')
X = df.iloc[: , :-2]

#Building the Autoencoder.
input_dim = X.shape[1]  # 8
# Define input layer.
input_data = Input(shape=(X.shape[1],))
Dropout_1 = (Dropout(0.9))(input_data)
encoder_layer_2 = Dense(3072, activation="relu")(Dropout_1)
Dropout_2 = (Dropout(0.9))(encoder_layer_2)
# Define encoding layer.
encoder_layer_3 = Dense(256, activation="relu")(Dropout_2)
Dropout_3 = (Dropout(0.9))(encoder_layer_3)
encoder_layer_4 = Dense(3072, activation="relu")(Dropout_3)
# Define decoding layer.
Dropout_4 = (Dropout(0.9))(encoder_layer_4)
# Define decoding layer.
decoded = Dense(X.shape[1], activation='relu')(Dropout_4)
# Create the autoencoder model
autoencoder = Model(input_data, decoded)

#Compile the autoencoder model.
autoencoder.compile(optimizer='Adagrad',
                    loss='binary_crossentropy')

#Fit to train set, validate with dev set and save to hist_auto for plotting purposes.
hist_auto = autoencoder.fit(X, X,
                epochs=100,
                batch_size=16,
                validation_split=0.45)

# Bottleneck representation.
encoder = Model(input_data, encoder_layer_3)

#Reconstructing features by applying the autoencoder on these features.
X_minimization = pd.DataFrame(autoencoder.predict(X))
#Removing zero variance features.
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
#Fitting the zero variance filter on the reconstructed features. 
constant_filter.fit(X_minimization)
len(X_minimization.columns[constant_filter.get_support()])
constant_columns = [column for column in X_minimization.columns
                    if column not in X_minimization.columns[constant_filter.get_support()]]
for column in constant_columns:
    print(column)
    
#Applying the zero variance filter on the reconstructed features.
features = constant_filter.transform(X_minimization)

# Normalise.
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
features = pd.DataFrame(features)

#Finding gene names related to the new reconstructed features.
all_columns = []
for i in range(0, 24366):
  all_columns.append(i)
 
list_difference = []
for item in all_columns:
  if item not in constant_columns:
    list_difference.append(item)
list_difference = pd.DataFrame(list_difference)

df = pd.concat([features, event], axis=1, join='inner')

#Random split into train and test subsets.
msk = np.random.rand(len(df)) < 0.7
train_features = df[msk]
test_features = df[~msk]

#Applying Cox Regression.
from lifelines import CoxPHFitter
cph = CoxPHFitter( penalizer=0.000004)
cph.fit(train_features, duration_col = 'Duration', event_col = 'Events', step_size = 5, show_progress=True)
results = cph.summary

#Drop all columns from results except P-value column.
results = results.drop(columns=['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', '-log2(p)'])

#Build the heatmap for p-values for the most important features.
import seaborn as sns
ax = sns.heatmap(results, annot=True, fmt='g')

#Train and test concordance.
from lifelines.utils import concordance_index
print(concordance_index(train_features['Duration'], -cph.predict_partial_hazard(train_features), train_features['Events']))
from lifelines.utils import concordance_index
print(concordance_index(test_features['Duration'], -cph.predict_partial_hazard(test_features), test_features['Events']))
es=cph.predict_survival_function(test_features)

#Plotting the calibration curve.
from sklearn.calibration import calibration_curve 
plt.figure(figsize=(4, 4))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2) 
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated") 
ax1.plot([0, 1], [0, 1], "k:") 
probs = 1-np.array(cph.predict_survival_function(test_features).loc[355.2])
actual = test_features['Events'] 
fraction_of_positives, mean_predicted_value =calibration_curve(actual, probs, n_bins=10, normalize=False) 
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % ("CoxPH",)) 
ax1.plot(mean_predicted_value, fraction_of_positives, "s-") 
ax1.set_ylabel("Fraction of positives") 
ax1.set_ylim([-0.05, 1.05]) 
ax1.legend(loc="lower right") 
ax1.set_title('Calibration plots (reliability curve)')

#Plotting survival function for the first 10 samples.
surv = cph.predict_survival_function(test_features)
paart = test_features.iloc[:10, :]
cph.predict_survival_function(paart).plot()

#Random Survival Forest
#First step is preparing the data for Random Survival Forest.
T = test_features['Duration']
E = test_features['Events']
from sksurv.util import Surv
X_test = test_features
X_test.drop('Events', axis=1)
X_test.drop('Duration', axis=1)

#Construct a structured array of event indicator and observed time.
y_test = Surv.from_arrays(test_features['Events'], test_features['Duration'])
from sksurv.util import Surv
X_train = train_features
X_train.drop('Events', axis=1)
X_train.drop('Duration', axis=1)
y_train = Surv.from_arrays(train_features['Events'], train_features['Duration'])

#Applying Random Survival Forest (RSF)
from sksurv.ensemble import RandomSurvivalForest
rsf = RandomSurvivalForest(n_estimators=50,
                           min_samples_split=7,
                           min_samples_leaf=10,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=124,
                           verbose=1)
rsf.fit(X_train, y_train)
Surv_Fun = rsf.predict_survival_function(X_train)
Cum_Hazard = rsf.predict_cumulative_hazard_function(X_train)

#Train and test concordance.
rsf.score(X_train, y_train)
rsf.score(X_test, y_test)

#Plotting calibration curve.
from sklearn.calibration import calibration_curve 
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2) 
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated") 
ax1.plot([0, 1], [0, 1], "k:")
Surv_Fun = np.transpose(pd.DataFrame(rsf.predict_survival_function(X_test))) 
probs = 1-np.array(Surv_Fun.loc[895])
actual = test_features['Events'] 
fraction_of_positives, mean_predicted_value =calibration_curve(actual, probs, n_bins=10, normalize=False) 
ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % ("RSF",), color='blue') 
ax1.plot(mean_predicted_value, fraction_of_positives, "s-") 
ax1.set_ylabel("Fraction of positives") 
ax1.set_ylim([-0.05, 1.05]) 
ax1.legend(loc="lower right") 
ax1.set_title('Calibration plots (reliability curve)')

# Predicting the weights for each feature.
results1 = rsf.summary
feature_names = X_test.head(n =172)
feature_names = X_test.columns.tolist()
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rsf, n_iter=15, random_state=1)
perm.fit(X_test, y_test)
# Needs a web-based interactive computing platform (for ex/ jupyter).
weights = eli5.show_weights(perm, feature_names=feature_names)

#Plotting Calibration Curve.
paart = X_test.iloc[0:10, :]
surv = rsf.predict_survival_function(paart)
for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.grid(True)
plt.legend()

#Calculating run time.
time = time.time() - start_time