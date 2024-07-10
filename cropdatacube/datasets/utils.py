
import numpy as np

from typing import Optional


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



def scale_data(img: np.ndarray, scaler: dict = None, function: str = 'standardization') -> np.ndarray:
    """
    Scales the input image data using the specified scaling function.

    Parameters
    ----------
    img : np.ndarray
        The image data to be scaled.
    scaler : dict, optional
        A dictionary containing the 'mean' and 'std' for standardization. Defaults to None.
    function : str, optional
        The scaling function to use. Options are 'standardization' or 'minmax'. Defaults to 'standardization'.

    Returns
    -------
    np.ndarray
        The scaled image data.
    """
    if function == 'standardization':
        scalefun = standard_scale
    elif function == 'minmax':
        scalefun = minmax_scale
    else:
        raise ValueError("Unsupported scaling function. Choose 'standardization' or 'minmax'.")

    if np.max(img) > 100:
        origimage = img.copy() / 255
    else:
        origimage = img.copy()
    
    if scaler is not None:
        
        assert len(scaler['mean']) == img.shape[0] # scaler must have the same image channels
        scaledimage = np.zeros(origimage.shape).astype(np.float64)
        for i in range(origimage.shape[0]):
            scaledimage[i,:] = scalefun(origimage[i,:], scaler['mean'][i], scaler['std'][i])
            
    else:
        scaledimage = origimage
    return scaledimage

def standard_scale(data: np.ndarray, meanval: float = None, stdval: float = None, navalue: float = 0) -> np.ndarray:
    """
    Standardizes the input data using the provided mean and standard deviation.

    Parameters
    ----------
    data : np.ndarray
        The data to be standardized.
    meanval : float, optional
        The mean value for standardization. Defaults to the mean of the data.
    stdval : float, optional
        The standard deviation value for standardization. Defaults to the standard deviation of the data.
    navalue : float, optional
        The value to be treated as NaN. Defaults to 0.

    Returns
    -------
    np.ndarray
        The standardized data.
    """
    if meanval is None:
        meanval = np.nanmean(data)
    if stdval is None:
        stdval = np.nanstd(data)
    if navalue == 0:
        datac1 = data.copy().astype(np.float64)
        ## mask na
        datac1[datac1 == navalue] = np.nan
        dasc = (datac1-meanval)/stdval 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data-meanval)/stdval

    return dasc


def minmax_scale(data: np.ndarray, minval: float = None, maxval: float = None, navalue: float = 0) -> np.ndarray:
    """
    Scales the input data using min-max normalization.

    Parameters
    ----------
    data : np.ndarray
        The data to be scaled.
    minval : float, optional
        The minimum value for scaling. Defaults to the minimum of the data.
    maxval : float, optional
        The maximum value for scaling. Defaults to the maximum of the data.
    navalue : float, optional
        The value to be treated as NaN. Defaults to 0.

    Returns
    -------
    np.ndarray
        The min-max scaled data.
    """
    
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    if navalue == 0:
        ## mask na
        data[data == navalue] = np.nan
        dasc = (data - minval) / ((maxval - minval)) 
        dasc[np.isnan(dasc)] = navalue 
    else:
        dasc= (data - minval) / ((maxval - minval))
    
    return dasc