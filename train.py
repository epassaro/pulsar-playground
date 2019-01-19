""" This script trains models defined in models.py. If no arguments are passed it trains all available models. """

import sys
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals.joblib import dump
from pulsar_playground.utils import *
from pulsar_playground.models import *


if __name__ == '__main__':

    if disable_warnings:
        warnings.filterwarnings("ignore")

    # Read training set
    train = pd.read_csv("./pulsar_playground/dataset/train_set.csv")
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:, -1]

    #%% Preprocessing tasks
    is_sampled = ''
    if oversample:

        sm = SMOTE(random_state=42)
        X_cols = X_train.columns

        X_train, y_train = sm.fit_sample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=X_cols)
    
        is_sampled = '_smote'

    if scale:

        scaler = StandardScaler()
        scaler.fit_transform(X_train)

    # Train models
    model_selection= list(sys.argv[1:])

    if len(model_selection) == 0:
        model_selection = model_dict.keys()

    for k in model_selection:

        model = model_dict[k]
        output_file = './saved_models/' + k + is_sampled
    
        # RandomizedSearch is used for large parameter grids only
        if get_n_params(model[1]) <= n_iter:
            clf = GridSearchCV(model[0], model[1], scoring=scoring, cv=cv, n_jobs=n_jobs)
        
        else:
            clf = RandomizedSearchCV(model[0], model[1], scoring=scoring, cv=cv, n_iter=n_iter, n_jobs=n_jobs)
    
        clf.fit(X_train, y_train)

        # Save trained models: Keras (HDF5) or Scikit/XGBoost (joblib)
        if isinstance(model[0], KerasClassifier):
            clf.best_estimator_.model.save(output_file + '.h5')

        else:    
            dump(clf.best_estimator_, output_file + '.joblib')

        # Print best parameters
        print(str(k).upper(), 'best params: ', clf.best_params_)
    
    print('\nFinished.')
