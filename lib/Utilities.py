import math
import random
import numpy as np
import keras.backend as K
from scipy.io import loadmat


# Step_decay function (every 20 epoch decrease the learning rate)
def step_decay(epoch):
   initial_lrate = 1e-3   # Change the initial learning rate here
   drop = 0.1
   epochs_drop = 20.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate

# Brier loss function
def brier_multi(targets, probs):
    return K.mean(K.sum((probs - targets)**2, axis=1))

# Collect the data (from the mat file)
def data_collection(path):
   all_data = loadmat(path)
   All_X_Data = all_data['X_Data']
   All_Y_Data = all_data['Y_Data']
   All_Y_Data_Int = all_data['Y_Data_Int']
   All_Y_Data = np.squeeze(All_Y_Data)
   cellLine_label = all_data['cellLine_Label']
   cellLine_label = np.squeeze(cellLine_label)

   random_ind = np.arange(All_X_Data.shape[3])
   random.shuffle(random_ind)

   X_train = All_X_Data[:, :, :, random_ind[:round(0.8 * All_X_Data.shape[3])]]
   Y_train = All_Y_Data[random_ind[:round(0.8 * All_X_Data.shape[3])]]

   X_test = All_X_Data[:, :, :, random_ind[round(0.8 * All_X_Data.shape[3]):]]
   Y_test = All_Y_Data[random_ind[round(0.8 * All_X_Data.shape[3]):]]

   Y_train_Int = All_Y_Data_Int[random_ind[:round(0.8 * All_X_Data.shape[3])], :]
   Y_test_Int = All_Y_Data_Int[random_ind[round(0.8 * All_X_Data.shape[3]):], :]

   return X_train, Y_train, X_test, Y_test, Y_train_Int, Y_test_Int, All_X_Data, cellLine_label