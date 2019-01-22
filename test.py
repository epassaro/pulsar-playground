""" This script computes AUC score for all models in saved_models. """

import sys
import warnings
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from pulsar_playground.parameters import *


if __name__ == '__main__':

    if disable_warnings:
        warnings.filterwarnings("ignore")

    # Read test set
    test = pd.read_csv("./pulsar_playground/dataset/test_set.csv")
    X_test = test.iloc[:,:-1]
    y_test = test.iloc[:, -1]

    # Preprocessing tasks
    if scale:
        scaler = StandardScaler()
        scaler.fit_transform(X_test)

    # Compute AuROC
    pickled = glob("./saved_models/*.joblib")
    hdf5 = glob("./saved_models/*.h5")

    print('\n::: AREA UNDER ROC CURVE :::\n')
    for model in pickled:
    
        clf = load(model)
        y_pred_proba = clf.predict_proba(X_test)

        model_name = str(model).replace('saved_models/','').replace('./','').replace('.joblib','')
        print(model_name, '= %.3f' % roc_auc_score(y_test, y_pred_proba[:,1]))

    for model in hdf5:

        clf = load_model(model)
        y_pred_proba = clf.predict_proba(X_test)

        model_name = str(model).replace('saved_models/','').replace('./','').replace('.h5','')

        # KerasClassifier predict_proba output has only one row.
        print(model_name, '= %.3f' % roc_auc_score(y_test, y_pred_proba[:,0]))

    print()
