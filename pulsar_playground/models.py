""" Module for defining models based on parameters.py file. """
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
#from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from .parameters import *


def keras_model(n, m, input_dim):
    """ Function to build a sequential neural network.
    
    Parameters
    -------
    n : int
        Number of hidden layers (network width).
    m : int
        Number of units per layer (network height).
    input_dim: int
        Length of feature vector.
    """
    model = Sequential()

    model.add(Dense(m, input_dim=input_dim, kernel_initializer='uniform', activation='relu'))
    
    for i in range(n-1):
        model.add(Dense(m, kernel_initializer='uniform', activation='relu'))
    
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model_dict = {}
""" dictionary: Stores the available models. """
model_dict['knn'] = (KNeighborsClassifier(), knn_params)
model_dict['lgr'] = (LogisticRegression(), lgr_params)
model_dict['xgb'] = (XGBClassifier(), xgb_params)
model_dict['ann'] = (KerasClassifier(build_fn=keras_model), ann_params)
