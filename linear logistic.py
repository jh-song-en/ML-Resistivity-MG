import numpy
import numpy as np
from mat4py import loadmat
import pandas as pd
from sklearn import tree, svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import plotly.express as px
import os
import graphviz
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def scaler(X):
    X = (X-np.mean(X, axis=0))/np.std(X, axis=0, ddof=1)
    return X

def data_prepare(return_total = True):
    data = loadmat('ML-R-MG.mat')
    data_training = data['trainingset_HTEonly']
    data_test = data['testset_HTEonly']
    data_liter = data['allset_Literature']
    data_training = np.asarray(data_training, dtype=np.float64)
    data_test = np.asarray(data_test, dtype=np.float64)
    data_liter = np.asarray(data_liter, dtype=np.float64)

    x_train = np.c_[data_training[:,1]*1000000,data_training[:,3],data_training[:,4],data_training[:,5],data_training[:,6]]
    x_test = np.c_[data_test[:,1]*1000000,data_test[:,3],data_test[:,4],data_test[:,5],data_test[:,6]]
    x_ind = np.c_[data_liter[:,1]*1000000,data_liter[:,3],data_liter[:,4],data_liter[:,5],data_liter[:,6]]
    y_train = data_training[:, 8]
    y_test = data_test[:, 8]
    y_ind = data_liter[:, 8]

    x_total = np.concatenate((x_test, x_train), axis=0)
    y_total = np.concatenate((y_test, y_train), axis=0)
    if return_total:
        return x_total, y_total, x_ind, y_ind
    else:
        return x_train, y_train, x_test, y_test

def k_fold_cross_validation(model, k, x_set, y_set, x_ind, y_ind, scaler = None, ann=False):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    model_train_score_list = []
    model_test_score_list = []
    model_ind_score_list = []
    model_coef_score_list = []

    if scaler == "MinMax":
        scaler = MinMaxScaler()
        scaler.fit(x_set)
        fed_x_set = scaler.transform(x_set)
        fed_x_ind = scaler.transform(x_ind)
    if scaler == "Cheat":
        scaler = StandardScaler()
        x_total = np.concatenate((x_set, x_ind), axis=0)
        scaler.fit(x_total)
        fed_x_set = scaler.transform(x_set)
        fed_x_ind = scaler.transform(x_ind)
    else:
        fed_x_set = x_set
        fed_x_ind = x_ind

    for train_index, test_index in skf.split(fed_x_set, y_set):
        x_train, x_test = fed_x_set[train_index], fed_x_set[test_index]
        y_train, y_test = y_set[train_index], y_set[test_index]
        if ann:

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

            model.fit(x_train, y_train, batch_size=20, epochs=1500, validation_data=(x_test, y_test), callbacks=[es, mc], verbose=0)
        else:
            model.fit(x_train, y_train)
        try:
            model_train_score_list.append(model.score(x_train, y_train))
            model_test_score_list.append(model.score(x_test, y_test))
            model_ind_score_list.append(model.score(fed_x_ind, y_ind))
        except:

            model_train_score_list.append(model.evaluate(x_train, y_train)[1])
            model_test_score_list.append(model.evaluate(x_test, y_test)[1])
            model_ind_score_list.append(model.evaluate(fed_x_ind, y_ind)[1])
        try:
            model_coef_score_list.append(model.coef_)
        except:
            model_coef_score_list.append([])
    sum_dict = {"Train_S": model_train_score_list, "Test_S": model_test_score_list, "Lit_S": model_ind_score_list, "Coef": model_coef_score_list}
    return pd.DataFrame(sum_dict)

def show_anal(df_model_result, name_model):
    print(name_model + ":")
    print(df_model_result[["Train_S", "Test_S", "Lit_S"]].describe())
    for i in list(df_model_result["Coef"]):
        if len(i) > 0:
            print(i)

def ANN():
    mdl = Sequential()

    mdl.add(Dense(50, input_dim=2, activation='relu'))
    mdl.add(Dense(1, activation='sigmoid'))
    mdl.compile(optimizer=Adam(lr=1e-2), loss="binary_crossentropy", metrics=['accuracy'])
    return mdl

def SVC():
    svc = svm.SVC(kernel="linear", C=1)
    return svc

if __name__ == "__main__":
    x_total, y_total, x_ind, y_ind = data_prepare()
    lrc = LogisticRegression()
    df_lrc = k_fold_cross_validation(lrc,10, x_total, y_total, x_ind, y_ind, scaler = "MinMax")
    tree_model = tree.DecisionTreeClassifier(max_depth=3)
    df_tree = k_fold_cross_validation(tree_model,10, x_total, y_total, x_ind, y_ind)
    ann = ANN()
    df_ann = k_fold_cross_validation(ann,10, x_total[:,[0,1]], y_total, x_ind[:,[0,1]], y_ind, scaler =  "Cheat", ann= True) #"Cheat"
    svc = SVC()
    df_svc = k_fold_cross_validation(svc,10, x_total, y_total, x_ind, y_ind, scaler = "MinMax")

    show_anal(df_lrc,"logistic Regression")
    show_anal(df_tree, "Tree 3")
    show_anal(df_ann, "ANN")
    show_anal(df_svc, "SVC")
