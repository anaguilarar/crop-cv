import os
import json
import tqdm
import random
import numpy as np
from itertools import compress, combinations
from typing import List, Optional
from .mls_functions import CVRegression
from .utils import select_columns

import pandas as pd
import copy
from sklearn.feature_selection import SequentialFeatureSelector


def save_options(opt, path, verbose = False):
        
    json_object = json.dumps(opt.__dict__, indent=4)

    with open(path, "w") as outfile:
        outfile.write(json_object)  
    
    print('saved in: {}', path)



class FeatureSelection():
    """
    A class for performing feature selection through combinations of features.

    Attributes
    ----------
    minfeatures : int
        Minimum number of features to start with.
    nmax_features : int
        Maximum number of features available for selection.
    """
    def __init__(self, nmax_features:int ,startswith: int = 3) -> None:
        """
        Initialize the FeatureSelection class.

        Parameters
        ----------
        nmax_features : int
            Maximum number of features available for selection.
        startswith : int, optional
            Minimum number of features to start with, by default 3.
        """
        
        self.minfeatures = startswith
        self.nmax_features = nmax_features
        
    def channels_combinations(self, nfeatures: int = 3, channel_names: Optional[List[str]] = None, shuffle: bool = False) -> List[List[str]]:
        """
        Generate unique combinations of channel names.

        Parameters
        ----------
        nfeatures : int, optional
            Number of features in each combination, by default 3.
        channel_names : list of str, optional
            Names of the channels to be combined, by default None.
        shuffle : bool, optional
            Whether to shuffle the combinations, by default False.

        Returns
        -------
        list of list of str
            List of unique combinations of channel names.
        """
        
        candidate_feature_indices = self._get_combinations(nfeatures)
        fninputs = [list(np.array(channel_names)[i]) for i in candidate_feature_indices]
        uniquefeat =[]
        for i in fninputs:
            i.sort()
            if i not in uniquefeat:
                uniquefeat.append(i)
        if shuffle:
            random.shuffle(uniquefeat)
            
        return uniquefeat
    
    def _combination_fixing_index(self, fixedindex: List[int], nfeatures: int = 3) -> List[List[int]]:
        """
        Generate combinations of features fixing some indices.

        Parameters
        ----------
        fixedindex : list of int
            Indices to be fixed in each combination.
        nfeatures : int, optional
            Total number of features in each combination, by default 3.

        Returns
        -------
        list of list of int
            List of combinations with fixed indices.
        """
        
        assert len(fixedindex) < nfeatures
        
        tmpind = fixedindex.copy()
        possiblecombs = self._get_combinations(nfeatures = nfeatures-(len(fixedindex)))
        tmpind.sort()
        
        onlythese = ~np.zeros(shape=len(possiblecombs), dtype=bool)
        for z in range(len(possiblecombs)):
            onlythese[z] = (onlythese[z] in [j in tmpind for j in possiblecombs[z]])
        candidate_feature_indices = np.flatnonzero(~onlythese)
        
        return [fixedindex + possiblecombs[i] for i in candidate_feature_indices]
        
        
    def _get_combinations(self, nfeatures: int = 3) -> List[List[int]]:
        """
        Generate all possible combinations of features.

        Parameters
        ----------
        nfeatures : int, optional
            Number of features in each combination, by default 3.

        Returns
        -------
        list of list of int
            List of all possible combinations of feature indices.
        """
        seqfeatures = list(range(self.nmax_features))
        combinationsvars = sum([list(map(list, combinations(seqfeatures, i))) 
                                    for i in range(len(seqfeatures) +  1)], [])
        
        combperrep = [i for i in combinationsvars if len(i) == nfeatures]
        
        return combperrep
    


class FeateruSelectorMethods(CVRegression):

    def reset_models(self):
        self.base_estimator = copy.deepcopy(self._raw_model)
        self._base_estimator = copy.deepcopy(self._raw_model)
        self.trained_models = {}

    def __init__(self, x, y, mlmodel, model_name=None, ids=None, val_perc=None, test_perc=None, seed=123, shuffle=True, testids_fixed=None) -> None:
        super().__init__(x, y, mlmodel, model_name, ids, val_perc, test_perc, seed, shuffle, testids_fixed)
        self.variables_names = self.X.columns.values
        self._raw_model = copy.deepcopy(self._base_estimator)

    def most_important_variables(self, kfolds = None, verbose = True):
        self.reset_models()
        self.cv_fit(kfolds, verbose = verbose)

        if self.model_name == 'rf':
            featureimportance = []
            for k in range(len(self.trained_models['rf'])):
                importances = self.trained_models['rf'][k][1].best_estimator_.feature_importances_
                featureimportance.append(pd.Series(importances, index=self.variables_names))

        return featureimportance

    def sfsfit_variableselector(self, nfeatures = None, kfolds = None):
        self.reset_models()
        
        self.base_estimator = SequentialFeatureSelector(self._base_estimator, n_features_to_select = nfeatures)
        self._base_estimator = copy.deepcopy(self.base_estimator)

        #self.model_name = self.model_name + '_sfs'
        self.cv_fit(kfolds)
        impvars = []
        variables = np.array(self.variables_names)
        for i in range(len(self.trained_models[self.model_name])):
            impvars += list(variables[self.trained_models[self.model_name][i].get_support()])

        values, counts = np.unique(impvars, return_counts=True)
        variables = pd.DataFrame({'features':values, 'frequency': counts})

        return variables

    def wrappper_exhaustive_search(self, kfolds = None, 
            checkpoint = None, onlynfeatures = None, verbose = False, filename = None):
        self.reset_models()

        if filename:
            filename = (filename + '_{}.csv').format(self.model_name)
        else:
            filename = '_{}.csv'.format(self.model_name)

        self._rawxdata = copy.deepcopy(self.X)
        combinationsvars = sum([list(map(list, combinations(self.variables_names, i))) 
                                    for i in range(len(self.variables_names) + 1)], [])
        modelsresults = []
        print('initial variables: ', np.unique(self.variables_names))
        if onlynfeatures is None:
            for n_feat in range(1,(len(np.unique(self.variables_names))+1)):
                print('*'*10, n_feat)
                combperrep = [i for i in combinationsvars if len(i) == n_feat]
                for columnsoi in combperrep:
                    
                    df_perquery = select_columns(self.X,columnsoi)
                    self.X = df_perquery
                    self.cv_fit(kfolds, verbose=verbose)
                    y_pred = self.cv_prediction()
                    
                    tablemetrics = pd.concat(self.cv_metrics_score(y_pred))
                    
                    tablemetrics["features"] = '-'.join(columnsoi)
                    tablemetrics["model"] = self.model_name
                    tablemetrics['n_features'] =  n_feat
                    modelsresults.append(tablemetrics.reset_index())
                    
                    self.X = self._rawxdata
                    if verbose:
                        print("{} R squared: {:.3f}".format('-'.join(columnsoi),tablemetrics['r2'].mean()))

                if checkpoint:
                    if not os.path.exists("fs_results"):
                        os.mkdir("fs_results")
                    
                    print('*'*10, 'checkpoint')
                    pd.concat(modelsresults).reset_index().to_csv(os.path.join('fs_results',filename))

        else:
            n_feat= onlynfeatures
            combperrep = [i for i in combinationsvars if len(i) == n_feat]
            for columnsoi in combperrep:
                print(columnsoi)
                df_perquery = select_columns(self.X,columnsoi)
                self.X = df_perquery
                self.cv_fit(kfolds)
                y_pred = self.cv_prediction()
                tablemetrics = self.cv_metrics_score(y_pred)[0]
                tablemetrics["features"] = '-'.join(columnsoi)
                tablemetrics["model"] = self.model_name
                tablemetrics['n_features'] =  n_feat
                modelsresults.append(tablemetrics)
                self.X = self._rawxdata
                if verbose:
                    print("{} R squared: {:.3f}".format('-'.join(columnsoi),tablemetrics['r2'].mean()))
                
        return modelsresults
