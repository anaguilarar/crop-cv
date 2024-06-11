

import json
from PIL import Image
import os
import re
import random

import numpy as np
import pandas as pd

from pathlib import Path
import pandas as pd



#from sklearn import linear_model


    
def assign_valuestoimg(data, height, width, na_indexes=None):
    ids_notnan = np.arange(height *
                           width)

    climg = np.zeros(height *
                     width, dtype='f')

    if na_indexes is not None:
        if len(na_indexes)>0:
            ids_notnan = np.delete(ids_notnan, na_indexes, axis=0)

    climg[list(ids_notnan)] = data
    #
    return climg.reshape(height, width)


def mask_usingGeometry(shp, xr_data):
    import rasterio.features
    ## converto to json
    jsonshpdict = shp.to_json()
    jsonshp = [feature["geometry"]
               for feature in json.loads(jsonshpdict)["features"]]

    # create mask
    return rasterio.features.geometry_mask(jsonshp,
                                           out_shape=(len(xr_data.y),
                                                      len(xr_data.x)),
                                           transform=xr_data.attrs['transform'],
                                           invert=True)


def get_maskmltfeatures(shp, sr_data, features_list, featurename='id'):
    if isinstance(features_list, list):

        if len(features_list) > 1:
            shpmask = []
            for i in features_list:
                shp_subset = shp[shp[featurename] == i]
                shpmask.append(mask_usingGeometry(shp_subset,
                                                  sr_data))

            shpmaskdef = shpmask[0]
            for i in range(1, len(shpmask)):
                shpmaskdef = np.logical_or(shpmaskdef, shpmask[i])

            return shpmaskdef


def get_maskmltoptions(dataarray, conditions_list):
    boolean_list = np.nan
    if isinstance(conditions_list, list):

        if len(conditions_list) > 1:
            boolean_list = dataarray == conditions_list[0]
            for i in range(1, len(conditions_list)):
                boolean_list = np.logical_or(boolean_list,
                                             (dataarray == conditions_list[i]))
        else:
            print("this function requires more than one filter condition")

    else:
        print('input is not a list')

    return boolean_list

def minmax_scale(data, minval = None, maxval = None):
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    return (data - minval) / ((maxval - minval))


def data_standarization(values, meanval = None, stdval = None):
    if meanval is None:
        meanval = np.nanmean(values)
    if stdval is None:
        stdval = np.nanstd(values)
    
    return (values - meanval)/stdval



def changenodatatonan(data, nodata=0):
    datac = data.copy()
    if len(datac.shape) == 3:
        for i in range(datac.shape[0]):
            datac[i][datac[i] == nodata] = np.nan

    return datac


def get_nan_idsfromarray(nparray):
    ids = []
    for i in range(nparray.shape[0]):
        ids.append(np.argwhere(
            np.isnan(nparray[i])).flatten())

    # ids = list(chain.from_iterable(ids))
    ids = list(np.concatenate(ids).flat)

    return np.unique(ids)


def from_xarray_2array(xrdata, bands, normalize = False):
    data_list = []
    for i in bands:
        banddata = xrdata[i].data
        banddata[banddata == xrdata.attrs['nodata']] = np.nan
        if normalize:
            banddata = (banddata *255)/ np.nanmax(banddata)

        data_list.append(banddata)

    return np.array(data_list)

def from_xarray_2list(xrdata, bands, normalize = False):
    data_list = []
    for i in bands:
        banddata = xrdata[i].data
        banddata[banddata == xrdata.attrs['nodata']] = np.nan
        if normalize:
            banddata = (banddata *255)/ np.nanmax(banddata)

        data_list.append(banddata)

    return data_list


def from_xarray_2_rgbimage(xarraydata,
                           bands=None,
                           export_as_jpg=False,
                           ouputpath=None, 
                           normalize = True,
                           newsize = None, verbose = False):


    if ouputpath is None:
        ouputpath = "image.jpg"
        directory = ""
    else:
        directory = os.path.dirname(ouputpath)

    if bands is None:
        bands = np.array(list(xarraydata.keys()))[0:3]

    data_tile = from_xarray_2array(xarraydata, bands, normalize)

    if data_tile.shape[0] == 3:
        data_tile = np.moveaxis(data_tile, 0, -1)

    image = Image.fromarray(data_tile.astype(np.uint8), 'RGB')
    if newsize is not None:
        image = image.resize(newsize)

    if export_as_jpg:
        Path(directory).mkdir(parents=True, exist_ok=True)

        if not ouputpath.endswith(".jpg"):
            ouputpath = ouputpath + ".jpg"

        image.save(ouputpath)
        if verbose:
            print("Image saved: {}".format(ouputpath))

    return image


def from_xarray_to_table(xrdata, nodataval=None,
                         remove_nan=True, features_names=None):
    if features_names is None:
        npdata = np.array([xrdata[i].data
                           for i in list(xrdata.keys())])
    else:
        npdata = np.array([xrdata[i].data
                           for i in features_names])

    if nodataval is not None:
        npdata = changenodatatonan(npdata,
                                   nodataval)

    # reshape to nlayers x nelements
    npdata = npdata.reshape(npdata.shape[0],
                            npdata.shape[1] * npdata.shape[2])

    idsnan = get_nan_idsfromarray(npdata)

    if remove_nan:
        if len(idsnan)>0:
            npdata = np.delete(npdata.T, idsnan, axis=0)
        else:
            npdata = npdata.T
    return [npdata, idsnan]


def resize_3dnparray( array,new_size=512):

    if new_size>array.shape[1] and new_size>array.shape[0]:
        resimg = []
        for i in range(array.shape[0]):
            tmp = array[i].copy()
            tmp = np.hstack([tmp, np.zeros([array.shape[1], (new_size-array.shape[2])])])
            resimg.append(np.vstack([tmp, np.zeros([(new_size-array.shape[1]), new_size])]))
    
    else:
        if new_size>array.shape[1]:
            resimg = []
            for i in range(array.shape[0]):
                tmp = array[i].copy()
                resimg.append(np.vstack([tmp, np.zeros([(new_size-array.shape[1]), new_size])]))


        if new_size>array.shape[2]:
            resimg = []
            for i in range(array.shape[0]):
                tmp = array[i].copy()
                resimg.append(np.hstack([tmp, np.zeros([new_size, (new_size-array.shape[2])])]))

    return np.array(resimg)

def summary_xrbyquantiles(xrdata, quantiles = [.25,0.5,0.75], idcolum = 'date'):

    df = xrdata.to_dataframe()
    df = df.groupby(idcolum).quantile(quantiles)
    if 'spatial_ref' in df.columns:
        df = df.drop('spatial_ref',axis = 1)
    df = df.reset_index()

    df['idt'] = 0
    df['id'] = df[idcolum].astype(str) + '_' + df['level_1'].astype(str)
    
    dflist = []
    for i in list(xrdata.keys()):
        dftemp = df.pivot(index='idt', columns='id', values=i).reset_index()
        dftemp = dftemp.drop(['idt'], axis = 1)

        dftemp.columns = i + '_d_' + dftemp.columns
        dflist.append(dftemp)

    return pd.concat(dflist, axis=1)
    
def get_vi_ts(df, npattern = ['ndvi']):

    tsdata = df.copy()
    for i in npattern:
        tsdata = tsdata.filter(regex=i)

    return np.expand_dims(tsdata.values,2)
"""

def get_linear_coefficients_fromtwoarrays(data, reference, verbose = True):
    
    dffit = pd.DataFrame({
        'x':data,
        'y':reference
    }).dropna()

    regr = linear_model.LinearRegression()
    regr.fit(np.expand_dims(dffit.x.values, axis = 1),
            np.expand_dims(dffit.y.values, axis = 1))
    if verbose:
        print("{} + X * {}".format(np.squeeze(regr.intercept_), np.squeeze(regr.coef_)))

    return np.squeeze(regr.intercept_), np.squeeze(regr.coef_)
"""


def from_long_towide(data, indexname, values_columnname, metrics_colname):

    widephenomycs = []
    for name in np.unique(data[metrics_colname].values):

        ssdata = data.loc[data[metrics_colname] == name]
        ssdatawide = ssdata.copy().reset_index()
        ssdatawide['idx'] = ssdatawide.groupby([indexname]).cumcount()
        ssdatawide['idx'] = np.unique(ssdatawide.metric.values) + '_t' + ssdatawide.idx.astype(str)
        
        widephenomycs.append(ssdatawide.pivot(
        index=indexname,columns='idx',values=values_columnname).reset_index())
    
    dfconcatenated = widephenomycs[0]
    for i in range(1,len(widephenomycs)):
        dfconcatenated = pd.merge(dfconcatenated,widephenomycs[i], how="left", on=[indexname])

    return dfconcatenated.reset_index()


def transform_listarrays(values, varchanels = None, scaler = None, scalertype = 'standarization'):
    
    if varchanels is None:
        varchanels = list(range(len(values)))
    if scalertype == 'standarization':
        if scaler is None:
            scaler = {chan:[np.nanmean(values[i]),
                            np.nanstd(values[i])] for i, chan in enumerate(varchanels)}
        fun = data_standarization
    elif scalertype == 'normalization':
        if scaler is None:
            scaler = {chan:[np.nanmin(values[i]),
                            np.nanmax(values[i])] for i, chan in enumerate(varchanels)}
        fun = data_normalization
    
    else:
        raise ValueError('{} is not an available option')
    
    valueschan = {}
    for i, channel in enumerate(varchanels):
        if channel in list(scaler.keys()):
            val1, val2 = scaler[channel]
            scaleddata = fun(values[i], val1, val2)
            valueschan[channel] = scaleddata
    
    return valueschan    


def customdict_transformation(customdict, scaler, scalertype = 'standarization'):
    """scale customdict

    Args:
        customdict (dict): custom dict
        scaler (dict): dictionary that contains the scalar values per channel. 
                       e.g. for example to normalize the red channel you will provide min and max values {'red': [1,255]}  
        scalertype (str, optional): string to mention if 'standarization' or 'normalization' is gonna be applied. Defaults to 'standarization'.

    Returns:
        xrarray: xrarraytransformed
    """

    varchanels = list(customdict['variables'].keys())
    values =[customdict['variables'][i] for i in varchanels]
    trvalues = transform_listarrays(values, varchanels = varchanels, scaler = scaler, scalertype =scalertype)
    for chan in list(trvalues.keys()):
        customdict['variables'][chan] = trvalues[chan]
        
    return customdict

