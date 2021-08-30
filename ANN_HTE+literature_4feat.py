import numpy
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def scaler(X):
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0, ddof=1)
    return X

# Load training/test datasets as matrix that includes all features and output
# (already divided into two sets randomly 20% selected test data)
# 0-absolute resistivity                1-modified resistivity
# 2-average atomic radii                3-atomic mismatch
# 4-heat of mixing                      5-average electronegativity
# 6-electronegativity difference        7-number of atom
# 8-Amorphous/Crystalline label (1/0)
data = loadmat('ML-R-MG.mat')
data_training = data['trainingset_HTEwithLiter']
data_test = data['testset_HTEwithLiter']
data_training = np.asarray(data_training, dtype=np.float64)
data_test = np.asarray(data_test, dtype=np.float64)

# Scaling input features on the total domain (training+test)
X = scaler(np.vstack((data_training[:, :8], data_test[:, :8])))

# Define training and test input, and their output as well
# Input example below can be varied on the specific utilization of features as you want
x_train = np.c_[X[:566,1],X[:566,3],X[:566,4],X[:566,5]]
x_test = np.c_[X[566:,1],X[566:,3],X[566:,4],X[566:,5]]
y_train = data_training[:, 8]
y_test = data_test[:, 8]

# Reshaping dataset as for network flow
x_train = x_train.reshape(566, 4)
x_test = x_test.reshape(142, 4)
y_train = y_train.reshape(566,1)
y_test = y_test.reshape(142,1)

# Early stopping the model training by validation loss
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Set learning rate in Adam optimizer (this value has been already optimized)
opt = Adam(lr=1e-2)

# Artificial neural networks of 50 neurons on 1-hidden layer (this has been already optimized)
mdl = Sequential()
mdl.add(Dense(50, input_dim=4, activation='relu'))
mdl.add(Dense(1, activation='sigmoid'))
mdl.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])
hst = mdl.fit(x_train, y_train, batch_size=20, epochs=1500,
            validation_data=(x_test, y_test), callbacks=[es, mc])

# Test accuracy at the end
print('-------------------------------------------------------------------------------------------------------------')
print("\r Test accuracy : %.4f" % (mdl.evaluate(x_test, y_test)[1])*100)