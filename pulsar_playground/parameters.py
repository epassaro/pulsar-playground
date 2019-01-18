""" Parameters for preprocessing and fine-tuning models. """
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


####################################################### 
#                    Miscellaneous                    #
#######################################################

disable_warnings = True
""" bool: Disable warnings. """


####################################################### 
#               Feature transformation                #
#######################################################

scale = True 
""" bool: Standarize features with StandardScaler. """

oversample = True
""" bool: Use SMOTE to fix class imbalance. """

rotate = False
""" bool: Use PCA. """

n_components = None 
""" integer: Number of features to keep if "rotate" is True. """


####################################################### 
#              Grid/RandomizedSearchCV                #
#######################################################

cv = 3              
""" integer: Number of cross-validation folds. """

scoring = 'recall'
""" string: Cross validation scoring method. """

n_jobs = 1
""" integer: Number of CPU threads to use with GridSearchCV or RandomizedSearchCV. """

n_iter = 100
""" integer: Max number of iterations for RandomizedSearchCV. """


####################################################### 
#                 Model parameters                    #
#######################################################

knn_params = dict( n_neighbors = range(3,12),
                   weights = ['uniform', 'distance'], )
""" dictionary: Parameter grid for KNeighborsClassifier.
        Please refer to Scikit Learn's documentation for more information. """ 

lgr_params = dict( penalty = ['l1','l2'],
                   C = np.arange(0.35, 0.45, 0.01), 
                   class_weight = [None, 'balanced'],
                   solver= ['liblinear'],
                   max_iter = [200], )
""" dictionary: Parameter grid for LogisticRegression. 
        Please refer to Scikit Learn's documentation for more information. """ 

xgb_params = dict( n_estimators = [200, 400],
                   learning_rate = [0.001], 
                   max_depth = [3], 
                   reg_alpha = [0.0001], 
                   gamma = [0.1], )
""" dictionary: Parameter grid for XGBoostClassifier.
        Please refer to the XGBoost API documentation for more information. """ 

ann_params = dict( n = np.arange(8, 20, 2),
                   m = np.arange(8, 20, 2),
                   input_dim = [8],
                   epochs = [200],
                   batch_size = [100], 
                   verbose = [0], 
                   callbacks = [[EarlyStopping(monitor='acc', patience=5, mode='auto')]], )
""" dictionary: Parameter grid for KerasClassifier. 
        If "rotate" is True then "input_dim" should match "n_components". Otherwise must be equal to number of features.
        Please refer to Keras documentation for more information. """