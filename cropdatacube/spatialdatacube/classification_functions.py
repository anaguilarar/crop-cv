from sklearn.preprocessing import MinMaxScaler

import random
from sklearn.cross_decomposition import PLSRegression

import numpy as np
import pandas as pd

import xarray
from .data_processing import from_xarray_to_table
from .data_processing import assign_valuestoimg



def img_rf_classification(xrdata, model, ml_features):
    idvarsmodel = [list(xrdata.keys()).index(i) for i in ml_features]

    if len(idvarsmodel) == len(ml_features):
        npdata, idsnan = from_xarray_to_table(xrdata,
                                                              nodataval=xrdata.attrs['nodata'],
                                                              features_names=ml_features)

        # model prediction
        ml_predicition = model.predict(npdata)

        # organize data as image
        height = xrdata.dims['y']
        width = xrdata.dims['x']

        return assign_valuestoimg(ml_predicition,
                                                  height,
                                                  width, idsnan)



def cluster_3dxarray(xrdata, cluster_dict):

    listnames = list(xrdata.keys())
    listdims = list(xrdata.dims.keys())

    npdata2dclean, idsnan = from_xarray_to_table(xrdata, 
                                                 nodataval= xrdata.attrs['nodata'])
    
    npdata2dclean = pd.DataFrame(npdata2dclean, columns = listnames)
    
    if 'scale_model' in list(cluster_dict.keys()):
        npdata2dclean = cluster_dict['scale_model'].transform(npdata2dclean)
    if 'pca_model' in list(cluster_dict.keys()):
        npdata2dclean = cluster_dict['pca_model'].transform(npdata2dclean)

    dataimage = cluster_dict['kmeans_model'].predict(npdata2dclean)
    xrsingle = xarray.DataArray(
            assign_valuestoimg(dataimage+1, xrdata.dims[listdims[0]],
                                            xrdata.dims[listdims[1]], 
                                            idsnan))
    xrsingle.name = 'cluster'
    
    return xrsingle



def cluster_4dxarray(xrdata, 
                     cl_dict,
                     cluster_value = None, 
                     only_thesedates = None,
                     name4d = 'date'):
    """
    function to mask a xarray multitemporal data using a pretrained cluster algorithm

    Parameters:
    ----------
    xarraydata: xarray
        data that contains the multi-temporal imagery data
    cl_dict: dict
        dictionary with the cluster model
    cluster_value: int
        cluster index that will be used as mask
    only_thesedates: list, optional
        scefy the time index if only those dates will be masked

    Return:
    ---------
    [an xarray filtered, the mask]
        
    """
    
    xrtobemasked = xrdata.copy()

    dim1size = range(xrtobemasked.dims[name4d])

    imgfilteredperdate = []
    maskslist = []
    tpmasked = []
    
    for i in dim1size:
        xrsingle = xrtobemasked.isel({name4d : i}).copy()
        
        if only_thesedates is not None:
            if i in only_thesedates:
            
                xrclusterlayer = cluster_3dxarray(
                    xrsingle[cl_dict['variable_names']].copy(), cl_dict)
                
                xrsingle = xrsingle.where(
                    xrclusterlayer.values != cluster_value,np.nan)
                maskslist.append(xrclusterlayer)
                tpmasked.append(i)
            
        else:
            xrclusterlayer = cluster_3dxarray(
                xrsingle[cl_dict['variable_names']].copy(), cl_dict)
            
            xrsingle = xrsingle.where(
                xrclusterlayer.values != cluster_value,np.nan)
            maskslist.append(xrclusterlayer)
            tpmasked.append(i)
              
        imgfilteredperdate.append(xrsingle)
    
    if len(imgfilteredperdate)>0:
        #name4d = list(xrdata.dims.keys())[0]

        mltxarray = xarray.concat(imgfilteredperdate, dim=name4d)
        mltxarray[name4d] = xrdata[name4d].values
    
    if len(maskslist)>0:
        masksxr = xarray.concat(maskslist, dim=name4d)
        masksxr[name4d] = xrdata[name4d].values[tpmasked]

    return mltxarray, masksxr