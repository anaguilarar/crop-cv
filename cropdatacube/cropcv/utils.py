
import numpy as np
import random
import os

from itertools import combinations
from typing import Optional

def perform(fun, *args):
    return fun(*args)

def perform_kwargs(fun, **kwargs):
    return fun(**kwargs)

def binary_classification_transform(scores: np.ndarray, 
                                    threshold: float, 
                                    comparing_with_zero: bool = True, 
                                    min_threshold: Optional[float] = None) -> np.ndarray:
    """
    Transforms continuous score data into binary categories based on a given threshold. 
    Scores equal to zero can be handled specifically, and an optional minimum threshold can be defined for further classification refinement.

    Parameters
    ----------
    scores : np.ndarray
        Array of score values.
    threshold : float
        The threshold above which scores are classified as 1.
    comparing_with_zero : bool, optional
        If True, scores exactly equal to zero are treated separately, default is True.
    min_threshold : Optional[float], optional
        An optional lower bound to classify scores. Scores below this threshold are set to nan unless they are zero, by default None.

    Returns
    -------
    np.ndarray
        A binary array where scores above the `threshold` are 1, and all others are 0. If `min_threshold` is set,
        scores below this value are set to nan. If `comparing_with_zero` is True, scores exactly equal to zero are handled separately.

    Notes
    -----
    The function uses numpy for logical operations to classify the scores efficiently.
    """
    scorecate = np.zeros(scores.shape)
    scorecate[scores>=threshold] = 1
    if comparing_with_zero:
        cerovals = scores == 0.0
        scorevals = scores >= threshold
        
        scorecate = np.zeros(scores.shape)
        scorecate[scorevals] = 1
        scorecate[np.logical_not(np.logical_or(cerovals,scorevals))] = np.nan
    
    if min_threshold is not None:
        cerovals = scores <= min_threshold
        scorevals = scores >= threshold
        
        scorecate = np.zeros(scores.shape)
        scorecate[scorevals] = 1
        scorecate[np.logical_not(np.logical_or(cerovals,scorevals))] = np.nan
    
    return scorecate



def summarise_trasstring(values):
    
    if type (values) ==  list:
        paramsnames = '_'.join([str(j) 
        for j in values])
    else:
        paramsnames = values

    return '{}'.format(
            paramsnames
        )


class FeatureSelection():
    
    @staticmethod
    def sorting_features(listfeatures):
        uniquefeat =[]
        for i in listfeatures:
            i.sort()
            if i not in uniquefeat:
                uniquefeat.append(i)
        
        return uniquefeat
    
    
    def channels_combinations_fixing_positions(self, fixed_indices, channel_names):
        candidate_feature_indices = self._combination_fixing_index(fixed_indices,
                                                                   len(fixed_indices)+1)

        candidate_feature_indices

        fninputs = [list(np.array(channel_names)[i]) for i in candidate_feature_indices]
        
        fninputs = self.sorting_features(fninputs)
            
        return fninputs
        
    
    def channels_combinations(self, nfeatures = 3, channel_names = None):
        candidate_feature_indices = self._get_combinations(nfeatures)
        fninputs = [list(np.array(channel_names)[i]) for i in candidate_feature_indices]
        
        fninputs = self.sorting_features(fninputs)
            
        return fninputs
    
    def _combination_fixing_index(self, fixedindex, nfeatures = 3):

        assert len(fixedindex) < nfeatures
        
        tmpind = fixedindex.copy()
        possiblecombs = self._get_combinations(nfeatures = nfeatures-(len(fixedindex)))
        tmpind.sort()
        
        onlythese = ~np.zeros(shape=len(possiblecombs), dtype=bool)
        for z in range(len(possiblecombs)):
            onlythese[z] = (onlythese[z] in [j in tmpind for j in possiblecombs[z]])
        candidate_feature_indices = np.flatnonzero(~onlythese)
        
        return [fixedindex + possiblecombs[i] for i in candidate_feature_indices]
        
        
    def _get_combinations(self, nfeatures = 3):
        seqfeatures = list(range(self.nmax_features))
        combinationsvars = sum([list(map(list, combinations(seqfeatures, i))) 
                                    for i in range(len(seqfeatures) +  1)], [])
        
        combperrep = [i for i in combinationsvars if len(i) == nfeatures]
        
        return combperrep
    
    def __init__(self, nmax_features) -> None:
        
        self.nmax_features = nmax_features
        