
from .general import MSVEGETATION_INDEX
import re
import numpy as np
import pickle
import os

import pandas as pd


def calculate_vi_fromarray(arraydata, variable_names,vi='ndvi', expression='(nir - green)/(nir + green)', label=None, navalues = None, overwrite = False):
    """
    Function to calculate vegetation indices given an equation and a multi-channels data array

    Args:
        arraydata (numpy array): multi-channel data array
        variable_names (list): list of the array channels names
        vi (str, optional): which is the name of the vegetation index that the user want to calculate. Defaults to 'ndvi'.
        expression (str, optional): vegetation index equation that makes reference to the channel names. Defaults to '(nir - green)/(nir + green)'.
        label (str, optional): if the vegetation index will have another name. Defaults to None.
        navalues (float, optional): numerical value which for non values. Defaults to None.
        overwrite (bool, optional): if the vegetation index is inside of the current channel names would you like to still calculate de index. Defaults to False.

    Raises:
        ValueError: Raises an error if the equation variables names are not in the provided channels names

    Returns:
        numpy array
    """
    
    if expression is None and vi in list(MSVEGETATION_INDEX.keys()):
        expression = MSVEGETATION_INDEX[vi]

    # modify expresion finding varnames
    symbolstoremove = ['*','-','+','/',')','.','(',' ','[',']']
    test = expression
    for c in symbolstoremove:
        test = test.replace(c, '-')

    test = re.sub('\d', '-', test)
    varnames = [i for i in np.unique(np.array(test.split('-'))) if i != '']
    
    for i, varname in enumerate(varnames):
        if varname in variable_names:
            exp = (['listvar[{}]'.format(i), varname])
            expression = expression.replace(exp[1], exp[0])
        else:
            raise ValueError('there is not a variable named as {}'.format(varname))

    listvar = []
    
    
    if vi not in variable_names or overwrite:

        for i, varname in enumerate(varnames):
            if varname in variable_names:
                pos = [j for j in range(len(variable_names)) if variable_names[j] == varname][0]

                varvalue = arraydata[pos]
                if navalues:
                    varvalue[varvalue == navalues] = np.nan
                listvar.append(varvalue)
        
        vidata = eval(expression)
            
        if label is None:
            label = vi
            
    else:
        vidata = None
        print("the VI {} was calculated before {}".format(vi, variable_names))
    

    return vidata, label


def get_data_from_dict(data, onlythesechannels = None):
            
        dataasarray = []
        channelsnames = list(data.variables.keys())
        
        if onlythesechannels is not None:
            channelstouse = [i for i in onlythesechannels if i in channelsnames]
        else:
            channelstouse = channelsnames
        for chan in channelstouse:
            dataperchannel = data['variables'][chan] 
            dataasarray.append(dataperchannel)

        return np.array(dataasarray)
    
    
