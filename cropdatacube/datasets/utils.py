
import numpy as np

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