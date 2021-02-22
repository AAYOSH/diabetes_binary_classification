import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pandas.plotting import scatter_matrix
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix



def create_model(n_hidden1,n_input,n_output):
    model = keras.Sequential()
    # INPUT
    model.add(keras.Input(shape = (n_inputs,)))
    # CAMADA DO MEIO 
    model.add(keras.layers.Dense(n_hidden1, activation='relu'))
    # CAMADA DE SAIDA # Sigmoid activation function best suited to binary classification
    model.add(keras.layers.Dense(n_outputs, activation='sigmoid'))
    
    return model


if __name__ == '__main__':

    df = pd.read_csv('../data/dados_diabetes.csv',sep = ',')

    # ====================================================================
    # SCALING DATASET 
    scaler = MinMaxScaler()
    scaler.fit(df)
    Scaled = scaler.transform(df)
    # =====================================================================

    # APPLYING A STRATIFIED HOLDOUT CROSS-VALIDATION
    X_train,X_test,y_train,y_test = train_test_split(Scaled[:,:-1],Scaled[:,-1],\
                                                 train_size = 0.8,test_size = 0.2,stratify = Scaled[:,-1])
    
    n_inputs = 8 # number of atributes
    n_outputs = 1

    #==========================================================================
    # searching for best number of neurons in the hidden layer
    test_loss_v = []
    test_acc_v = []
    neurons_v = np.arange(1,5000,20)
    for n in neurons_v:
        model = create_model(n,n_inputs,n_outputs)
        model.compile(optimizer='adam', #> adam as the best optmizer 
                    loss='binary_crossentropy',
                    metrics=['binary_accuracy'])
        model.fit(X_train, y_train, epochs=14,batch_size = 6) # number of epoch reduced to 14 due to loop iteration
        test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2)
        test_loss_v.append(test_loss)
        test_acc_v.append(test_acc)


    # =============================================================================
    # BEST N FOUND: 2661
    # =============================================================================
    # applying the model using the history of .fit method to evaluate our model also in validation dataset
    n = 2661
    model = create_model(n)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    history = model.fit(X_train, y_train, epochs = 70, batch_size = 6, verbose = 1, validation_data = (X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2)

    #==================================================================================
    # find the best number of epochs to stop the training(see images)
    n = 2661
    model = create_model(n)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    history = model.fit(X_train, y_train, epochs = 46, batch_size = 6, verbose = 1, validation_data = (X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test,y_test,verbose=2)


    #====================================================================================
    #predicting
    y_pred = model.predict_classes(X_test)
    print(confusion_matrix(y_test, y_pred))