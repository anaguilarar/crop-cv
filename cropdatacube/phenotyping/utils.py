import numpy as np
import pandas as pd
from typing import Optional

from typing import Tuple, List

def calculate_quantiles(nparray, quantiles = [0.25,0.5,0.75]):
    """
    Calculates specified quantile values for each 1D array along the first dimension of a numpy array.

    Parameters
    ----------
    nparray : np.ndarray
        The input numpy array from which quantiles are calculated. It expects an array order of HW. if a 3D array is given
        the array order must be CHW.
    quantiles : List[float], optional
        The list of quantiles to calculate, by default [0.25, 0.5, 0.75].

    Returns
    -------
    List[Dict[float, float]]
        A list of dictionaries, with each dictionary containing the quantiles for each 1D array within the numpy array.
        The keys are the quantile values requested, and the values are the calculated quantiles.

    Examples
    --------
    >>> import numpy as np
    >>> nparray = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    >>> calculate_quantiles(nparray)
    [{0.25: 15.0, 0.5: 20.0, 0.75: 25.0}, {0.25: 45.0, 0.5: 50.0, 0.75: 55.0}, {0.25: 75.0, 0.5: 80.0, 0.75: 85.0}]
    """
    
    listqvalues = []
    if len(nparray.shape)>2:
        for i in range(nparray.shape[0]):
            datq = {q : np.nanquantile(nparray[i].flatten(),q = q) for q in quantiles}
            listqvalues.append(datq)
    else:
        listqvalues = {q : np.nanquantile(nparray.flatten(),q = q) for q in quantiles}
            
    return listqvalues


def from_dict_to_df(dict_losses):
    channdf = []
    for channel_name, value in dict_losses.items():
        df_channel = pd.DataFrame(value, index=[channel_name])
        channdf.append(df_channel)

    df = pd.concat(channdf, axis=0).reset_index()
    return df


def from_quantiles_dict_to_df(quantiles_dict, 
                              idvalue: Optional[str] = None) -> pd.DataFrame:
    """
    Converts a dictionary with quantiles data to a pandas DataFrame, adjusts variable names, 
    and restructures the DataFrame to have quantiles as columns and an optional ID value.

    Parameters
    ----------
    quantiles_dict : Dict
        The dictionary containing quantiles data, with keys as variable names.
    idvalue : Optional[str], optional
        An identifier value to be added to all rows in the output DataFrame, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the quantiles data restructured, having an ID value (if provided),
        and quantiles as columns for each variable.
    """
    
    idvalue = '0' if idvalue is None else idvalue
    channdf = []
    for channel_name, value in quantiles_dict.items():
        df_channel = pd.DataFrame(value, index=[idvalue])
        df_channel.columns = df_channel.columns.map(lambda x: '{}_{}'.format(channel_name, x).replace('.',''))
        channdf.append(df_channel)

    df_wide = pd.concat(channdf, axis=1).reset_index()
    
    df_wide = df_wide.rename(columns={'index': 'id'})

    return df_wide



def line_intersection(line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Compute the intersection point of two lines.
    
    Parameters
    ----------
    line1 : List[Tuple[float, float]]
        Coordinates of the first line defined by two points (x1, y1) and (x2, y2).
    line2 : List[Tuple[float, float]]
        Coordinates of the second line defined by two points (x3, y3) and (x4, y4).
    
    Returns
    -------
    Tuple[float, float]
        The (x, y) coordinates of the intersection point.
    
    Raises
    ------
    Exception
        If the lines do not intersect (i.e., they are parallel).
    
    Notes
    -----
    This function is based on the method described at:
    https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    """
    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('Lines do not intersect.')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def perpendicular_points_to_line(linecoords: List[float], points: np.ndarray, diff_factor: float = 0.1) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Get perpendicular points to a line.

    Parameters
    ----------
    linecoords : List[float]
        Coordinates of the line (x1, y1, x2, y2).
    points : np.ndarray
        Array of point coordinates with shape (N, 2).
    diff_factor : float, optional
        A tolerance factor for determining perpendicularity. Default is 0.1.

    Returns
    -------
    List[Tuple[float, float]]
        The (x, y) coordinates of the perpendicular points.
    """
    
    x1,y1,x2,y2 = linecoords
    slope_points = (y2-y1)/(x2-x1)
    
    perpendicular_coords = []
    for i in range(points.shape[0]):
        pi0 = points[i]
        for j in range(i, points.shape[0]):
            pj1 = points[j]
            if (pi0[0]-pj1[0]) !=0:
                is_perpendicular = slope_points * (pi0[1]-pj1[1])/(pi0[0]-pj1[0])
                if -1.0 - diff_factor <= is_perpendicular <= -1.0 + diff_factor:
                    perpendicular_coords.append([pi0, pj1])

    return perpendicular_coords
