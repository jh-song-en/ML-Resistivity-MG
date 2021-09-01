import numpy
import numpy as np
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
# Literature data is also loaded for independent test (transferability test as itself)
# 0-absolute resistivity                1-modified resistivity
# 2-average atomic radii                3-atomic mismatch
# 4-heat of mixing                      5-average electronegativity
# 6-electronegativity difference        7-number of atom
# 8-Amorphous/Crystalline label (1/0)
data = loadmat('ML-R-MG.mat')
data_training = data['trainingset_HTEonly']
data_test = data['testset_HTEonly']
data_liter = data['allset_Literature']
data_training = np.asarray(data_training, dtype=np.float64)
data_test = np.asarray(data_test, dtype=np.float64)
data_liter = np.asarray(data_liter, dtype=np.float64)

# Scaling input features on the total domain (training+test)
# (literature data would be scaled within the known scaling domain)
X1 = scaler(np.vstack((data_training[:, :8], data_test[:, :8])))
X2 = scaler(np.vstack((data_training[:, :8], data_test[:, :8], data_liter[:, :8])))

# Define training and test input, and their output as well
# Input example below can be varied on the specific utilization of features as you want
x_train = np.c_[X1[:242,1],X1[:242,3],X1[:242,5]]
x_test = np.c_[X1[242:,1],X1[242:,3],X1[242:,5]]
x_ind = np.c_[X2[302:,1],X2[302:,3],X2[302:,5]]
y_train = data_training[:, 8]
y_test = data_test[:, 8]
y_ind = data_liter[:, 8]

# Reshaping dataset as for network flow
x_train = x_train.reshape(242, 3)
x_test = x_test.reshape(60, 3)
x_ind = x_ind.reshape(406, 3)
y_train = y_train.reshape(242,1)
y_test = y_test.reshape(60,1)
y_ind = y_ind.reshape(406, 1)


# Early stopping the model training by validation loss
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

# Set learning rate in Adam optimizer (this value has been already optimized)
opt = Adam(lr=1e-2)

# Artificial neural networks of 50 neurons on 1-hidden layer (this has been already optimized)
mdl = Sequential()
mdl.add(Dense(50, input_dim=3, activation='relu'))
mdl.add(Dense(1, activation='sigmoid'))
mdl.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])
hst = mdl.fit(x_train, y_train, batch_size=20, epochs=1500,
            validation_data=(x_test, y_test), callbacks=[es, mc])

# Test accuracy
print('-------------------------------------------------------------------------------------------------------------')
print("\r Test accuracy : %.4f" % (mdl.evaluate(x_test, y_test)[1])*100)

# Test accuracy of literature data for transferability
print('-------------------------------------------------------------------------------------------------------------')
print("\r Accuracy on literature : %.4f" % (mdl.evaluate(x_ind, y_ind)[1])*100)
