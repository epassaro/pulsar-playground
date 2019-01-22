""" This script trains models defined in models.py. If no arguments are passed it trains all available models. """
import sys
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals.joblib import dump
from pulsar_playground.utils import *
from pulsar_playground.models import *


if __name__ == '__main__':

    # Check user input
    model_selection= list(sys.argv[1:])

    if len(model_selection) == 0:
        print("\nUsage: python train.py <model_1> <model_2> ...\n")
        sys.exit(0)

    best_params = []
    for m in model_selection:
        if m not in model_dict.keys():
            print('\nNo model named \'%s\'.' % m)
            get_models()
            sys.exit(1)

    if disable_warnings:
        warnings.filterwarnings("ignore")
    
    # Read training set
    train = pd.read_csv("./pulsar_playground/dataset/train_set.csv")
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:, -1]

    # Preprocessing tasks
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
    for m in model_selection:

        model = model_dict[m]
    
        # RandomizedSearch is used for large parameter grids only
        if get_n_params(model[1]) <= n_iter:
            clf = GridSearchCV(model[0], model[1], **searchargs)
        
        else:
            clf = RandomizedSearchCV(model[0], model[1], n_iter=n_iter, **searchargs)
    
        clf.fit(X_train, y_train)

        # Print best parameters
        best_params.append(clf.best_params_)
   
        # Train again with best params and all available data
        clf_final = model[0].set_params(**clf.best_params_)
        clf_final.fit(X_train, y_train)

        # Save trained models: Keras (HDF5) or Scikit/XGBoost (joblib)
        output_file = './saved_models/' + m + is_sampled

        if isinstance(model[0], KerasClassifier):
            clf_final.model.save(output_file + '.h5')

        else:    
            dump(clf_final, output_file + '.joblib')


    print('\n\n::: BEST PARAMETERS :::\n')
    for i in range(len(model_selection)):
        print(str(model_selection[i]), ':', best_params[i])
    
    print('\nFinished.')
