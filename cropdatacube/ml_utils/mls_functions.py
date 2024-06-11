from sklearn.multioutput import RegressorChain
from timeit import default_timer as timer
from datetime import timedelta

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.linear_model import Lasso, Ridge, RidgeClassifier


from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.utils.fixes import loguniform
from sklearn.svm import SVR, SVC


import math

import random

from ..datasets.datasplit import SplitIds, split_dataintotwo, retrieve_datawithids

import os
import copy
import tqdm





def pca_transform(data,
                  variancemin=0.5,
                  export_pca=False,
                  sample = 'all',
                  seed = 123):
    """

    :param data: numpy array
    :param varianzemin: numeric
    :param export_pca: boolean
    :return: dictionary
    """
    if sample == "all":
        datatotrain = data
    elif sample < data.shape[0]:
        random.seed(seed)
        random_indices = random.sample(range(data.shape[0]), sample)
        datatotrain = data[random_indices]

    pca = PCA()
    pca.fit_transform(datatotrain)
    # define the number of components through
    ncomponets = np.max(np.argwhere((pca.explained_variance_ * 100) > variancemin)) + 1

    #print("calculating pca with {} components".format(ncomponets))
    # calculate new pca
    pca = PCA(n_components=ncomponets).fit(datatotrain)
    data = pca.transform(data)
    # export data
    output = {'pca_transformed': data}

    if export_pca:
        output['pca_model'] = pca

    return output


def kmeans_images(data: np.ndarray, 
                  nclusters: int,
                  scale: str = "minmax",
                  nrndsample: int = "all",
                  seed: int = 123,
                  pca: bool = True,
                  export_pca: bool = False,
                  eigmin: float = 0.3,
                  verbose: bool = False) -> dict:
    """
    Perform K-means clustering on image data, optionally using PCA for dimensionality reduction first.

    Parameters
    ----------
    data : np.ndarray
        Image data for clustering, expected shape (C, H*W) where C is channels and H*W are the spatial dimensions flattened.
    nclusters : int
        Number of clusters for K-means.
    scale : str, optional
        Type of scaling to apply, default is 'minmax'.
    nrndsample : int or str, optional
        Number of random samples to use, or 'all' to use all data, default is 'all'.
    seed : int, optional
        Random seed for reproducibility, default is 123.
    pca : bool, optional
        Whether to perform PCA before clustering, default is True.
    export_pca : bool, optional
        Whether to export the PCA model, default is False.
    eigmin : float, optional
        Minimum explained variance ratio for PCA components to keep, default is 0.3.
    verbose : bool, optional
        If True, print additional details during processing, default is False.

    Returns
    -------
    dict
        A dictionary containing clustering results and models including labels, K-means model, scaling model, and possibly PCA model.
    """
    if scale == "minmax":
        scaler = MinMaxScaler().fit(data)

    scaleddata = scaler.transform(data)
    if verbose:
        print("scale done!")

    if pca:
        pcaresults = pca_transform(scaleddata, eigmin, export_pca, nrndsample)
        scaleddata = pcaresults['pca_transformed']

    if nrndsample == "all":
        datatotrain = scaleddata
    elif nrndsample < scaleddata.shape[0]:
        random.seed(seed)
        random_indices = random.sample(range(scaleddata.shape[0]), nrndsample)
        datatotrain = scaleddata[random_indices]

    if verbose:
        print("kmeans training using a {} x {} matrix".format(datatotrain.shape[0],
                                                              datatotrain.shape[1]))

    kmeansclusters = KMeans(n_clusters=nclusters,
                            random_state=seed).fit(datatotrain)
    clusters = kmeansclusters.predict(scaleddata)
    output = {
        'labels': clusters,
        'kmeans_model': kmeansclusters,
        'scale_model': scaler,
        'pca_model': np.nan
    }
    if export_pca:
        output['pca_model'] = pcaresults['pca_model']

    return output

def set_model(model_name = 'pls',
              scaler = 'standardscaler', 
              param_grid = None, 
              scale_data = True,
              cv = 5, 
              nworkers = -1):
    
    """
    function to set a shallow learning model for regression, this is a sklearn function which first will scale the data, then will 
    do a gridsearch to find the best hyperparameters

    Parameters:
    ----------
    model_name: str
        which is the model that will be used
        {'pls': Partial least square,
         'svr_radial': support vector machine with radial kernel,
         'svr_linear': support vector machine with linear kernel,
         'rf': Random Forest,
         'lasso', 'ridge', 'default': 'pls'}
    scaler: str
        which data scaler will be applied
        {'minmax', 'standardscaler', default: 'standardscaler'}
    param_grid: dict, optional
        grid parameters used for hyperparameters gird searching
        
    scale_data: boolean, optional
        use scaler in the model
    cv: int
        k-folds for cross-validation
    nworkers: int
        set the number of workers that will be used for parallel process

    Returns:
    ---------
    pipemodel

    """
    if scaler == 'minmax':
        scl = MinMaxScaler()
    if scaler == 'standardscaler':
        scl = StandardScaler()

    if model_name == 'pls':
        if param_grid is None:
            rdcomps = np.linspace(start = 1, stop = 50, num = 30)
            param_grid = [{'n_components':np.unique([int(i) for i in rdcomps])}]

        mlmodel = GridSearchCV( PLSRegression(),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)

    if model_name == 'svr_linear':
        if param_grid is None:
            param_grid = {'C': loguniform.rvs(0.1, 1e3, size=20),
                          'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}

        ## model parameters
        mlmodel  = GridSearchCV(SVR(kernel='linear'),
                                          param_grid,
                                          cv=cv,
                                          n_jobs=nworkers)


    if model_name == 'svr_radial':
        if param_grid is None:
            param_grid = {'C': loguniform.rvs(0.1, 1e3, size=20),
                          'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}
        ## model parameters
        mlmodel  = GridSearchCV(SVR(kernel='rbf'),
                                          param_grid,
                                          cv=cv,
                                          n_jobs=nworkers)



    if model_name == 'xgb':
        if param_grid is None:
            param_grid = {
                    'min_child_weight': [1, 2, 4],
                    'gamma': [0.001,0.01,0.5, 1, 1.5, 2, 5],
                    'n_estimators': [100, 500],
                    'colsample_bytree': [0.7, 0.8],
                    'max_depth': [2,4,8,16,32],
                    'reg_alpha': [1.1, 1.2, 1.3],
                    'reg_lambda': [1.1, 1.2, 1.3],
                    'subsample': [0.7, 0.8, 0.9]
                    }

        xgbreg = xgb.XGBRegressor(
                        eval_metric="rmse",
                        random_state = 123
                )
        mlmodel  = RandomizedSearchCV(xgbreg,
                               param_grid,
                               cv=cv,
                               n_jobs=nworkers,
                               n_iter = 50)

       
    if model_name == 'rf':
        if param_grid is None:
            param_grid = { 
            'n_estimators': [300],
            'max_features': [0.15, 0.3, 0.4, 0.6],
            'max_depth' : [2,4,8,16,32],
            'min_samples_split' : [2,4,8],
            'max_samples': [0.7,0.9],
                        
            #'max_leaf_nodes': [50, 100, 200]
            #'criterion' :['gini', 'entropy']
            }
        mlmodel = GridSearchCV( RandomForestRegressor(random_state = 42),
                                param_grid,
                                cv=5,
                                n_jobs=-1)
    

    
    if model_name == 'lasso':
        if param_grid is None:
            alphas = np.logspace(-4, -0.5, 30)
            param_grid = [{"alpha": alphas}]
            
        mlmodel  = GridSearchCV(Lasso(random_state=0, max_iter=4000),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)

    if model_name == 'ridge':
        if param_grid is None:
            alphas = np.logspace(-4, -0.5, 30)
            param_grid = [{"alpha": alphas}]
            
        mlmodel  = GridSearchCV(Ridge(random_state=0, max_iter=4000),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
        
    
    if scale_data:
        pipelinemodel = Pipeline([('scaler', scl), (model_name, mlmodel)])
    else:
        pipelinemodel = Pipeline([(model_name, mlmodel)])


    return pipelinemodel

def set_classification_model(model_name = 'rf',
              scaler = 'standardscaler',
              scale_data = True,
              param_grid = None, 
              cv = 3, 
              nworkers = -1,
              seed = 123):
    
    """
    function to set a shallow learning model for classification, this is a sklearn function which first will scale the data, then will 
    do a gridsearch to find the best hyperparameters

    Parameters:
    ----------
    model_name: str
        which is the model that will be used
        {
         'svr_radial': support vector machine with radial kernel,
         
         'rf': Random Forest,
         'xgb': xtra gradient boosting}
    scaler: str
        which data scaler will be applied
        {'minmax', 'standardscaler', default: 'standardscaler'}
    param_grid: dict, optional
        grid parameters used for hyperparameters gird searching
    cv: int
        k-folds for cross-validation
    nworkers: int
        set the number of workers that will be used for parallel process

    Returns:
    ---------
    pipemodel

    """
    if scaler == 'minmax':
        scl = MinMaxScaler()
    if scaler == 'standardscaler':
        scl = StandardScaler()
    if model_name == 'ridge':
        if param_grid is None:
            alphas = np.logspace(-4, -0.5, 30)
            param_grid = [{"alpha": alphas}]
            
        mlmodel = GridSearchCV(RidgeClassifier(random_state=0, max_iter=4000),
                                param_grid,
                                cv=cv,
                                n_jobs=nworkers)
        

    if model_name == 'svc_radial':
        if param_grid is None:
            param_grid = {'C': loguniform.rvs(0.1, 1e3, size=20),
                          'gamma': loguniform.rvs(0.0001, 1e-1, size=20)}

        ## model parameters
        mlmodel  = GridSearchCV(SVC(kernel='rbf',
                                          random_state = seed),
                                          param_grid,
                                          cv=cv,
                                          n_jobs=nworkers)
    if model_name == 'xgb':
        if param_grid is None:
            param_grid = {
                    'min_child_weight': [1, 2, 4],
                    'gamma': [0.001,0.01,0.5, 1, 1.5, 2, 5],
                    'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 500, num = 3)],
                    'subsample':[i/10.0 for i in range(6,10)],
                    'colsample_bytree':[i/10.0 for i in range(6,10)],
                    'max_depth': [2,4,8,16,32],
                    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],

                    }

        xgbclass = xgb.XGBClassifier(objective="binary:logistic",
                        random_state = seed
                )
        if len(list(param_grid.keys())) > 0:
            gs_xgb  = RandomizedSearchCV(xgbclass,
                               param_grid,
                               cv=cv,
                               n_jobs=nworkers,
                               n_iter = 100,
                               random_state = seed)

            mlmodel = gs_xgb

        else:
            mlmodel = xgbclass
        
    if model_name == 'rf':
        if param_grid is None:

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 5)]
            # Number of features to consider at every split
            max_features = [0.3,0.6,0.9]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [2, 4,8]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            param_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}

        rf = RandomForestClassifier(random_state=seed)
        mlmodel = RandomizedSearchCV(estimator = rf, param_distributions = param_grid,
                                               n_iter = 100, cv = cv,
                                               verbose=0, random_state=seed, n_jobs = nworkers)
       

    if scale_data:
        pipelinemodel = Pipeline([('scaler', scl), (model_name, mlmodel)])
    else:
        pipelinemodel = Pipeline([(model_name, mlmodel)])


    return pipelinemodel
    

def check_real_predictionshapes(real, prediction):
    """
    Regarding the model, the outpur could have a different shape than the real data
    """
    if not len(real.shape) == len(prediction.shape):
        if real.shape > prediction.shape:
            real = np.squeeze(real)
        else:
            prediction = np.squeeze(prediction)
    
    return real, prediction

def check_arraytype(data, typeinput = 'target'):
    if type(data) != np.ndarray:
        data = data.to_numpy()

    if data.shape == 1 and typeinput == 'input':
        data = np.expand_dims(data,axis = 1)
    return data

def mae(real, prediction):
    real, prediction = check_real_predictionshapes(real, prediction)
    real, prediction = np.array(real), np.array(prediction)
    return np.mean(np.abs(real - prediction))
    
##https://www.sciencedirect.com/topics/engineering/root-mean-square-error
def rrmse(real, prediction):
    real, prediction = check_real_predictionshapes(real, prediction)
    EPSILON =  1e-10 ## avoid errors caused when dividing using 0
    return (np.sqrt(np.mean((real - prediction)**2)) / (np.mean(real) + EPSILON)) * 100


def get_eval_metrics(real, prediction):
    return (pd.DataFrame({
                'r2': [r2_score(y_true=real,
                                y_pred=prediction)],
                'rmse': [math.sqrt(mean_squared_error(y_true=real,
                                y_pred=prediction))],
                'rrmse': [rrmse(real=real,
                                prediction=prediction)],
                'mae': [mae(real=real,
                                prediction=prediction)]}))

class CVRegression(SplitIds):
    
    def __init__(self, x,y, mlmodel,model_name = None, ids=None, val_perc=None, test_perc=None, seed=123, shuffle=True, testids_fixed=None) -> None:
        #ids_length = None, ids = None,val_perc =None, test_perc = None,seed = 123, shuffle = True, testids_fixed = None
        super().__init__(ids_length = x.shape[0], ids = ids, val_perc = val_perc, test_perc = test_perc, seed = seed, shuffle = shuffle, testids_fixed= testids_fixed)
        #self.base_estimator = set_model(model_name, cv=gs_kfolds)
        self.model_name = "model" if model_name is None else model_name
        self.base_estimator = copy.deepcopy(mlmodel)
        self._base_estimator = copy.deepcopy(mlmodel)
        
        self.X = copy.deepcopy(x)
        self.Y = copy.deepcopy(y)
        self.variables_names = self.X.columns.values

    def cv_fit(self, kfolds = None, verbose = True):

        #if kfolds is None:
        trainedmodels = [0]*kfolds
        
        self.trained_models = {}

        for k in tqdm.tqdm(range(kfolds), disable= (not verbose)):
            tr_x, tr_y, _, _ = self.get_xyvaldata(self.X, self.Y, kfolds=kfolds, kifold = k, 
                                                  phase= 'training')
            tr_x = check_arraytype(tr_x, typeinput = 'input')
            tr_y = check_arraytype(tr_y)
                       
            m = self.base_estimator.fit(tr_x, tr_y)
            trainedmodels[k] = copy.deepcopy(self.base_estimator)
            self.base_estimator = copy.deepcopy(self._base_estimator)
        
        self.trained_models[self.model_name] = trainedmodels

    def cv_prediction(self):

        #model_name = list(self.trained_models.keys())[0] if model_name is None else model_name 

        nmodels = len(self.trained_models[self.model_name])
        Y = []
        for m in range(nmodels):
            _, _, val_x, _ = self.get_xyvaldata(self.X, 
                                               self.Y, 
                                               kfolds=nmodels, 
                                               kifold = m)
            val_x = check_arraytype(val_x, typeinput = 'input')
            pred = self.trained_models[self.model_name][m].predict(val_x)
            Y.append(np.squeeze(pred))
        
        return Y

    def cv_metrics_score(self, Y_pred):

        eval_metrics = []
        
        for i in range(len(Y_pred)):
            _, _, _, val_y = self.get_xyvaldata(self.X, 
                                               self.Y, 
                                               kfolds=len(Y_pred), 
                                               kifold = i)
            val_y = check_arraytype(val_y)                                   
            pdmetric = get_eval_metrics(val_y, Y_pred[i])               
            pdmetric['cv'] = [i]
            eval_metrics.append(pdmetric)
        
        return eval_metrics

    def get_xyvaldata(self, X, Y, kfolds=None, kifold = None, phase = 'training'):

        kifold = 0 if kifold is None else kifold
        
        if kfolds is None:
            
            if phase == "validation":
                tr_x, val_x = split_dataintotwo(X, 
                                            idsfirst = self.training_ids, 
                                            idssecond = self.test_ids)

                tr_y, val_y = split_dataintotwo(Y, 
                                            idsfirst = self.training_ids, 
                                            idssecond = self.test_ids)
            if phase == "training":
                tr_x = retrieve_datawithids(X, self.training_ids) 
                tr_y = retrieve_datawithids(Y, self.training_ids)
                val_x, val_y = None, None

        else:
            
            tr_x, val_x = split_dataintotwo(X, 
                                            idsfirst = self.kfolds(kfolds)[kifold][0], 
                                            idssecond = self.kfolds(kfolds)[kifold][1])

            tr_y, val_y = split_dataintotwo(Y, 
                                            idsfirst = self.kfolds(kfolds)[kifold][0], 
                                            idssecond = self.kfolds(kfolds)[kifold][1])

        return tr_x, tr_y, val_x, val_y

