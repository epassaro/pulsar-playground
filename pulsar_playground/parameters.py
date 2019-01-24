""" Parameters for preprocessing and fine-tuning models. """
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


####################################################### 
#                    Miscellaneous                    #
#######################################################

disable_warnings = True
""" bool: Disable warnings. """


####################################################### 
#                Preprocessing tasks                  #
#######################################################

scale = True 
""" bool: Standarize features with StandardScaler. """

oversample = True
""" bool: Use SMOTE to fix class imbalance. """


####################################################### 
#              Grid/RandomizedSearchCV                #
#######################################################

n_iter = 100
""" integer: Max number of iterations for RandomizedSearchCV. """

searchargs = dict( cv = 3, 
                   scoring = 'accuracy',
                   n_jobs = -1,
                   verbose = 2,
                 )
""" dictionary: Extra arguments for Grid/RandomSearchCV. """


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

xgb_params = dict( n_estimators = [400],
                   max_depth = [3],
                   min_child_weight = [3],
                   gamma = [5],
                   colsample_bytree = [0.8],
                   learning_rate = [0.01],
                   subsample = [1], )
""" dictionary: Parameter grid for XGBoostClassifier.
        Please refer to the XGBoost API documentation for more information. """ 

xgb_gpu_params = dict( tree_method = ['gpu_hist'],
                   predictor = ['cpu_predictor'],
                   n_estimators = [400],
                   max_depth = [7],
                   min_child_weight = [1],
                   gamma = [9],
                   learning_rate = [0.05],
                   colsample_bytree = [1.0],
                   subsample = [1.0], )
""" dictionary: Parameter grid for XGBoostClassifier (GPU).
        Please refer to the XGBoost API documentation for more information. """ 

ann_params = dict( n = [1, 2],
                   m = [12, 14],
                   input_dim = [8],
                   epochs = [10],
                   batch_size = [100],
                   drop_visible = [0.0],
                   drop_hidden = [0.0, 0.1, 0.2],
                   verbose = [0],
                   callbacks = [[EarlyStopping(monitor='acc', patience=3, mode='auto')]], )
""" dictionary: Parameter grid for KerasClassifier. 
        If "rotate" is True then "input_dim" should match "n_components". Otherwise must be equal to number of features.
        Please refer to Keras documentation for more information. """