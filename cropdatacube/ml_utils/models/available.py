
import numpy as np
import torch
import os
from io import BytesIO
import requests
from urllib.parse import urlparse

import zipfile

from .dl_architectures import maskrcnn_instance_segmentation_model


class AvailableModels:
    
    def _S3_models_weights(self):
        {'seeds_detection': "https://dlmodels-bucket.s3.ap-northeast-1.amazonaws.com/maskrcnn_rice_seeds_2.zip"}
    
    @property
    def _instance_segmentation(self):
        {'mask_rcnn_v2': maskrcnn_instance_segmentation_model}
    
    @property
    def _classification(self):
        
        ["cnn_transformer", "cnn3d_transformer"]
    



def filter_files_usingsuffix(filesinside, path, suffix = 'pt'):
    """
    function to pull a zip file from internet
    Parameters:
    --------
    path: str
        path that contains the files
    
    suffix: str
        use a string to filter the files that are inside the extracted folder
    
    Returrn:
    --------
    path to the file
    """

    fileinfolder = [i for i in filesinside if i.endswith(suffix)]
    print(fileinfolder)
    if len(fileinfolder)==1:
        wp = fileinfolder[0]
        wp = os.path.join(path, wp)
    else:
        raise ValueError("there is no files with this extension {}".format(suffix))
       
    return wp



def downloadzip(urlpath, foldername = 'models')-> None: 
    """
    function to pull a zip file from internet
    Parameters:
    --------
    urlpath: str
        url link which contian the file
    
    foldername: str
        the folder name in which the extracted file will be located
    
    Returrn:
    --------
    None
    """
    if foldername is None:
        foldername = ""

    if urlpath.startswith('http'):
        a = urlparse(urlpath)
        
        if not os.path.exists(os.path.join(foldername,os.path.basename(a.path))):
            req = requests.get(urlpath)

            with zipfile.ZipFile(BytesIO(req.content)) as zipobject:
                zipobject.extractall(foldername)
        
        else:
            zipobject = zipfile.ZipFile(os.path.join(foldername,os.path.basename(a.path)))
            if not os.path.exists(os.path.join(foldername,
                zipobject.filelist[0].filename)):
                with zipfile.ZipFile(os.path.basename(a.path)) as zipobject:
                    zipobject.extractall(foldername)

        return zipobject.namelist()

def check_weigth_path(path, suffix = 'h5', weights_path = 'weights'):
    """
    Check and retrieve the weight file path, downloading and extracting if necessary.

    Parameters
    ----------
    path : str
        The initial path to the weight file or zip archive.
    suffix : str, optional
        The expected file extension of the weight file (default is 'h5').
    weights_path : str, optional
        The directory to store the extracted weight files (default is 'weights').

    Returns
    -------
    str
        The path to the weight file with the specified suffix.
    
    Raises
    ------
    FileNotFoundError
        If the weight file with the specified suffix cannot be found.
    """
    if not path.endswith(suffix):
        
        if not os.path.exists(weights_path):
            filesinside = downloadzip(path, foldername = weights_path)
        else:
            filesinside = os.listdir(weights_path)
        
        path = filter_files_usingsuffix(filesinside, weights_path, suffix=suffix)
        if not path:
            raise FileNotFoundError(f"No file with suffix '{suffix}' found in '{weights_path}'.")
        
    return path

