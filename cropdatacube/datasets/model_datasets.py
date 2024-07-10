import copy
import json
import os
import inspect
import pandas as pd
import numpy as np
import time
import warnings

from typing import Tuple, Optional, List, Any

import torch
from torch.utils.data import Dataset


from .datasplit import SplitIdsClassification
from ..datacubepredictors.segmentation import SegmentationDataCube
from ..phenotyping.utils import calculate_quantiles, from_quantiles_dict_to_df
from ..spatialdatacube.datacube_processors import MultiDDataTransformer, DataCubeReader
from ..utils.general import split_filename

import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm


class TargetDataset():
    """
    A class to manage datasets containing target variables, with support for loading data from a CSV file and retrieving non-NaN target values.

    Parameters
    ----------
    phase : str
        Specifies the dataset phase, 'train' or 'validation'.
    path : str, optional
        The file path to the dataset CSV file. If not specified, no file will be loaded.
    target_key : str, optional
        The column name in the dataset that contains the target values.
    id_key : str, optional
        The column name that contains identifiers for each data point.
    tabletype : str, optional
        The type of data structure to load the data into, default is 'dataframe'.

    Attributes
    ----------
    train : bool
        Whether this dataset is for training.
    validation : bool
        Whether this dataset is for validation.
    file_path : str
        The path to the dataset file.
    target_label : str
        The column name for target values.
    ids_label : str
        The column name for data identifiers.
    _df : pd.DataFrame
        The loaded dataset as a DataFrame.
    """
    @property
    def df(self):
        if self._df is None:
            self._df = self.read_path(path = self.file_path)
            self._nnpos = self._non_nan_positions(self._df[self.target_label].values)
                        
        return self._df.loc[self._nnpos]
    
    
    def __init__(self, dataframe: pd.DataFrame = None, path:str=None, target_key:str = None, id_key:str = None, table_type = 'dataframe') -> None:

        self.target_label = target_key
        self.ids_label = id_key
        self._df = None
        self._reader_fun = None
        self._nnpos = None
        self.file_path = path
        self._reader_fun = pd.read_csv
        
        #self.target_transformation = parser.scaler_transformation
        if table_type == "dataframe":

            if path is not None:
                self._df = self.read_path(path = self.file_path) 
            elif dataframe is not None:
                self._df = dataframe
            
       #if tabletype == "dict":

    def read_path(self, path, reader = None):
        assert os.path.exists(path), f"Unable to use the specified path: {path}"
        if reader is None:
            reader = pd.read_csv
        
        return reader(path)        
    
    @staticmethod
    def _non_nan_positions(target):
        """
        Finds positions of non-NaN entries in the target column.

        Returns
        -------
        List[int]
            Indices of non-NaN entries in the target column.
        """
        #target = self._df[self.target_label].values
        return [i for i,value in enumerate(target) if not np.isnan(value)]
        
    
    def get_ids(self):
        """
        Retrieves the identifiers for data points with non-NaN target values.

        Returns
        -------
        List[int]
            A list of identifiers corresponding to non-NaN target values.
        """

        if self.ids_label in self.df.columns:
            ids = list(self._df[self.ids_label].values[self._nnpos])
        else:
            ids = list(range(self._nnpos))
        return ids
    
    
    def get_target(self):
        """
        Retrieves the target values that are non-NaN.

        Returns
        -------
        np.ndarray
            Array of non-NaN target values.
        """

        target = self._df[self.target_label].values[self._nnpos]    
        
        return target
    
    @property
    def ids_data(self):
        return self.get_ids()
    
    @property
    def target_data(self):
        return self.get_target()
    
    
class ClassificationTarget(TargetDataset, SplitIdsClassification):
    
    """
    Handles dataset operations for classification tasks including data retrieval and stratified data splitting.

    Inherits from:
    - TargetDataset for basic data handling.
    - SplitIdsClassification for stratified splitting of IDs based on target values.

    Methods
    -------
    split_data(cv=None, nkfolds=None)
        Splits the data into training and validation sets based on the provided cross-validation setup or number of folds.
    get_target_value(index)
        Retrieves the target value and corresponding ID for the specified index.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes the ClassificationTarget with data handling and splitting capabilities.

        Parameters are inherited from TargetDataset and SplitIdsClassification, including:
        - phase (str): Specifies if the dataset is used for training or validation.
        - path (str, optional): Path to the dataset CSV file.
        - target_name, id_colname, tabletype: Dataset specific configurations.
        - targetvalues (np.ndarray): Array of target classification values.
        - ids, val_perc, test_perc, seed, shuffle, testids_fixed, stratified: Data splitting parameters.
        """
        parameters = inspect.signature(TargetDataset.__init__).parameters
        #print('Dataset params: ', self.get_params_fromwargs(parameters,kwargs['DATASET']))
        
        TargetDataset.__init__(self, **self.get_params_fromwargs(parameters,kwargs['DATASET']))
        
        parameters = inspect.signature(SplitIdsClassification.__init__).parameters
        SplitIdsClassification.__init__(self, targetvalues=self.target_data, **self.get_params_fromwargs(parameters,kwargs['DATASPLIT']))
        self._idssubset = None
        self._valuesdata = None

   
    #def get_params_fromwargs(class)
    
    def split_data(self,  cv: Optional[int] = None, nkfolds: Optional[int] = None, phase = 'train', stratified_target = True) -> Tuple[List[Any], List[Any]]:
        """
        Splits the data for training or validation using either cross-validation indices or a number of folds.

        Parameters
        ----------
        cv : Optional[int]
            The specific cross-validation split index to use.
        nkfolds : Optional[int]
            Number of folds if using k-fold splitting.
        phase : str
            train or validation 
        stratified_target: bool
            stratifed split
        Returns
        -------
        Tuple[List[Any], List[Any]]
            A tuple containing lists of IDs and corresponding target data for the split.
        """
        if phase not in ['validation','train','test','all']:
            warnings.warn(f"{phase} must be either train or validation, phase was set as train")
            self.phase = 'train'
        else:
            self.phase = phase
        trdata, idsdata = None, None

        
        if  self._idssubset is None and self._valuesdata is None:
        # getting training or validation ids
            if self.phase == 'train':
                # TODO: CASE WHERE THE KFOLDS ARE NOT STRATIFIED
                if stratified_target:
                    idstosplit = self._get_new_stratified_ids(self._initial_tr_ids
                                                         ) if cv is None else self.stratified_kfolds(nkfolds)[cv][0] 
                else:
                    idstosplit = self._initial_tr_ids if cv is None else self.stratified_kfolds(nkfolds)[cv][0] 
                    
            elif self.phase == 'validation':
                if stratified_target:
                    idstosplit = self._get_new_stratified_ids(self._initial_test_ids    
                                                          ) if cv is None else self.stratified_kfolds(nkfolds)[cv][1]
                else:
                    idstosplit = self._initial_test_ids if cv is None else self.stratified_kfolds(nkfolds)[cv][1]
                    
            elif self.phase == 'test':
                idstosplit = self._get_new_stratified_ids(self._initial_test_ids) if stratified_target else self._initial_test_ids
            else:
                idstosplit = self._get_new_stratified_ids(list(self._initial_test_ids) + list(self._initial_tr_ids)
                                                          ) if stratified_target else list(self._initial_test_ids) + list(self._initial_tr_ids)
            
            idsdata = [self.ids_data[i] for i in idstosplit]
            trdata = [self.target_data[i] for i in idstosplit]
            
            self._idssubset = idsdata
            self._valuesdata = trdata
            
        return [idsdata, trdata]
    
    
    def get_target_value(self, index: int) -> Tuple[str, float]:
        """
        Retrieves the target value and its corresponding ID based on the provided index.

        Parameters
        ----------
        index : int
            Index for which to retrieve the target and ID.

        Returns
        -------
        Tuple[Any, Any]
            The ID and target value at the specified index.
        """
        return [self._idssubset[index], self._valuesdata[index]]
        
    @staticmethod
    def get_params_fromwargs(classparams,kwargs):
        """
        Extracts parameters for class initialization from arguments passed to the constructor.

        Parameters
        ----------
        classparams : Dict[str, Any]
            Parameters expected by the class constructor.
        kwargs : Dict[str, Any]
            Arguments provided to the constructor.

        Returns
        -------
        Dict[str, Any]
            Filtered dictionary of parameters applicable to the class constructor.
        """
        
        parameters = {key: value for key, value in classparams.items() if key not in ['self', 'args', 'kwargs']}
        parameterstarget = {key: kwargs.get(key) for key in kwargs.keys() if key in list(parameters.keys())}
        
        return parameterstarget 
    

class DataCubeTransformBase(DataCubeReader,SegmentationDataCube):
            
    def __init__(self, configuration_dict: dict)-> None : 

        import time
        self.confi = configuration_dict
        self.confi_datacube_transform = copy.deepcopy(configuration_dict['DATASETTRANSFORM'])

        self._mask_image = self.confi['DATASET'].get('mask_datacube',None)
        if self._mask_image:
            SegmentationDataCube.__init__(self)
        else:
            DataCubeReader.__init__(self)
        self._report_times = False
        self._cn = self.confi['DATASET']['feature_names'].split('-')
        self._scalar_values = self._load_scaler()
    
    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns
        -------
        int
            Total number of items.
        """
        
        return len(self._idssubset)
    
    def _check_filename(self, filename):
        """
        Check if the provided filename includes a directory path and update class variables accordingly.

        Parameters
        ----------
        filename : str
            The file name or path to split.

        Returns
        -------
        None
        """
        path, fn = split_filename(filename)
        self._tmppath = path
        self._file = fn

    def _split_data_in_traning_and_validation(self, cv = -1, nkfolds = 0, phase = None):
        self._datasplit_cv = None if cv == -1 else cv
        self._datasplit_nkfolds = None if nkfolds == 0 else nkfolds
        self._idssubset = None
        self._valuesdata = None
        
        return self.split_data(cv = self._datasplit_cv, nkfolds = self._datasplit_nkfolds, phase=phase)
        
    def _load_scaler(self):
        scaler_path = self.confi_datacube_transform.get('scaler_path',None)
        # load form path
        if scaler_path and os.path.exists(scaler_path):
            if os.path.exists(scaler_path):
                print(f'loaded scale values from : {scaler_path}')
                with open(scaler_path, 'rb') as file:
                    return json.load(file) 
        #load directly from configuration         
        elif 'scalar_values' in list(self.confi_datacube_transform.keys()):
            print(f'loaded from configuration file')
            return self.confi_datacube_transform.get('scalar_values',None)
            
        else:
            return None
            
            
    def get_datacube(self,data_path, 
                        channel_names = None, 
                        image_reduction  = None, 
                        transform_options = None,
                        mask_name = None) -> np.ndarray:

        #assert self._idssubset is not None, "Split the data first using _split_data_in_traning_and_validation function"
        channel_names = self._cn if channel_names is None else channel_names
        image_reduction = self.confi_datacube_transform['image_reduction'] if image_reduction is None else image_reduction
        transform_options = self.confi_datacube_transform['transform_options'] if transform_options is None else transform_options

        self._check_filename(data_path)
        self.read_individual_data(
                    path = self._tmppath, file=self._file,  dataformat = self.confi['DATASET']['input_array_order'])
        
        if self._mask_image:
            # if no mask name is provided then choose one randomly
            if mask_name is None:
                layernames = self.mask_layer_names(mask_suffix=self.confi['DATASET']['preffix_layername'])
                mask_name = np.random.choice(layernames)
                
            xrdata = self.clip_using_mask(mask_name=mask_name,mask_data=True, 
                                          padding= self.confi['DATASET']['padding'], paddingincm= ['padding_incm'])
        else:
            xrdata = self.xrdata
        ## activate tranform module
        
        datacube_metrics = MultiDDataTransformer(xrdata, 
                                            transformation_options =transform_options,
                                            channels=channel_names,
                                            scaler= self._scalar_values)

        datatransformed = datacube_metrics.get_transformed_image(min_area = self.confi_datacube_transform['minimun_area'], 
                                    image_reduction=image_reduction,
                                    #augmentation='raw',
                                    new_size= self.confi_datacube_transform['input_image_size'],
                                    rgb_for_color_space = self.confi_datacube_transform['rgb_channels_for_color_space'],
                                    rgb_for_illumination = self.confi_datacube_transform['rgb_channels_for_illumination'],
                                    scale_method = self.confi_datacube_transform['scale_method'],
                                    report_times=self._report_times)
        
        return datatransformed


class DataCubeModelBase(DataCubeTransformBase,ClassificationTarget):
            
    def __init__(self, configuration_dict: dict, split_in_init = True, stratified_target = True)-> None : 
        """
        Initializes the classification data handler with configurations.

        Parameters
        ----------
        configuration_dict : dict
            Configuration dictionary specifying operational parameters.
        split_in_init : bool
            Split in initialization
            
        """
        import time
        self.confi = configuration_dict
        #self.confi_datacube_transform = copy.deepcopy(configuration_dict['DATASETTRANSFORM'])
        self._report_times = False
        
        ClassificationTarget.__init__(self,**self.confi)
        
        DataCubeTransformBase.__init__(self, configuration_dict)
        
        self._report_times = False
        self._cn = self.confi['DATASET']['feature_names'].split('-')
        
        if split_in_init:
            self._split_data_in_traning_and_validation(cv = self.confi['DATASPLIT']["cv"], 
                                                       nkfolds = self.confi['DATASPLIT']["kfolds"],
                                                       phase=self.confi['DATASET']["phase"],
                                                       stratified_target=self.confi['DATASPLIT']["stratified_target"])
            
    
    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns
        -------
        int
            Total number of items.
        """
        
        return len(self._idssubset)
    
    def _split_data_in_traning_and_validation(self, cv = -1, nkfolds = 0, phase = None, stratified_target = True):
        self._datasplit_cv = None if cv == -1 else cv
        self._datasplit_nkfolds = None if nkfolds == 0 else nkfolds
        self._idssubset = None
        self._valuesdata = None
        
        return self.split_data(cv = self._datasplit_cv, nkfolds = self._datasplit_nkfolds, phase=phase, stratified_target = stratified_target)
        
            
    def get_data(self,index, channel_names = None, image_reduction  = None, 
                 transform_options = None, mask_name = None) -> Tuple[np.ndarray, float]:
        """_summary_

        Parameters
        ----------
            index (_type_): _description_
            channel_names (_type_, optional): _description_. Defaults to None.
            image_reduction (_type_, optional): _description_. Defaults to None.
            transform_options (_type_, optional): _description_. Defaults to None.
            mask_name (_type_, optional): _description_. Defaults to None.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and its corresponding target tensor.
        """
        
        assert self._idssubset is not None, "Split the data first using _split_data_in_traning_and_validation function"
        idimg, targetval = self.get_target_value(index)
        # get imagery data

        datatransformed = self.get_datacube(idimg,  
                                            channel_names = channel_names,
                                            image_reduction  = image_reduction, 
                                            transform_options = transform_options,
                                            mask_name = mask_name)
    
        
        return datatransformed, targetval


class ClassificationDLData(Dataset,DataCubeModelBase):
    
    def __init__(self, configuration_dict: dict, phase = 'train',cv = None, nkfolds = None, stratified = True)-> None : 
        """
        Initializes the data handler for DL models.

        Parameters
        ----------
        configuration_dict : dict
            Configuration dictionary specifying operational parameters.
        """
        
        #split_in_init = True if nkfolds is not None else False
            
        DataCubeModelBase.__init__(self,configuration_dict, split_in_init = False)
        self._split_data_in_traning_and_validation(cv = cv, nkfolds=nkfolds, phase = phase, stratified_target=stratified)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import time
        """
        Retrieves an item by its index for model training/testing.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and its corresponding target tensor.
        """
        datatransformed,targetval = self.get_data(index)
        
        imgtensor = torch.from_numpy(datatransformed.swapaxes(0,1)).float()
        
        targetval = np.array(targetval)
        targetten = torch.from_numpy(np.expand_dims(targetval, axis=0)).float()
        
        #if imgtensor.shape[1] == 1:
        #    imgtensor = torch.squeeze(imgtensor)
        imgtensor = torch.squeeze(imgtensor)
 
        return imgtensor, targetten
 
class DataSummarizer():
     
    def __init__(self) -> None:
        pass
     
    @staticmethod
    def _3d_array_summary(npdata,channel_names= None, channel_name: str = None, quantiles: List[float] = None, 
                          array_order ="DCHW", navalue = 0):
        """
        Summarizes 3D images into quantiles for a specific channel.

        Parameters
        ----------
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with quantile values for each depth slice of the 3D image.

        Raises
        ------
        ValueError
            If the dataset's array order is not 'DCHW', indicating depth, channel, height, width.
        """
        if array_order != "DCHW":
            raise ValueError('Currently implemented only for "DCHW" array order.')
        
        datasummary = []
        for i in range(npdata.shape[0]):
            xrdata2d = npdata[i]
            datasummary.append(DataSummarizer._2d_array_summary(xrdata2d, channel_names = channel_names,
                                                                      channel_name = channel_name, quantiles=quantiles))
        
        return datasummary
    
    @staticmethod
    def _2d_array_summary(npdata: np.ndarray, channel_names= None, channel_name = None, quantiles: List[float] = None, navalue = 0):
        """
        Summarizes 2D images into quantiles for a specific channel.

        Parameters
        ----------
        npdata :np.ndarray
            The array dataset to summarize.
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with calculated quantile values.

        Raises
        ------
        AssertionError
            If the specified channel name is not in the datacube's channel names.
        """
        if not quantiles:
            quantiles = [0.5]

        #assert channel_name in self._cn, "Channel name must be in the datacube."
        channel_name_i = channel_names.index(channel_name)
        channel_image = npdata[channel_name_i]
        if not navalue:
            channel_image[channel_image == navalue] = np.nan
            
        datasummary = calculate_quantiles(channel_image, quantiles=quantiles)
        
        return datasummary
    
    @staticmethod
    def summarise_into_quantiles(npdata: np.ndarray, channel_names: List[str], quantiles: List[float] = None,array_order = 'DCHW', navalue=0):
        """
        Summarizes the data of specified channels into quantiles for 2D and 3D images within the dataset.

        Parameters
        ----------
        channel_names : Optional[List[str]], optional
            A list of channel names to summarize. If None, all channels are used, by default None.
        quantiles : Optional[List[float]], optional
            A list of quantiles to calculate for each channel. If None, defaults to the median (0.5), by default None.

        Returns
        -------
        Dict[str, List[Dict[float, float]]]
            A dictionary where keys are channel names and values are lists of dictionaries, each containing calculated quantile values.
        """

        channel_data = {}
        ndims = len(npdata.shape)-1
        
        for channel_name in channel_names:
            if ndims == 2:
                channel_data[channel_name] = DataSummarizer._2d_array_summary(
                    npdata, channel_names = channel_names,channel_name=channel_name, quantiles= quantiles, navalue=navalue)
            if ndims == 3:
                channel_data[channel_name] = DataSummarizer._3d_array_summary(
                    npdata, channel_names = channel_names, channel_name=channel_name, quantiles= quantiles, array_order=array_order, navalue=navalue)
        
        return channel_data
 
class ClassificationMLData(DataCubeModelBase, DataSummarizer):
    def __init__(self, configuration_dict: dict, split_in_init: bool = True)-> None : 

        """
        Initializes the data handler for ML models.

        Parameters
        ----------
        configuration_dict : dict
            Configuration dictionary specifying operational parameters.
        split_in_init : bool
            Split data in initialization
        """
        DataCubeModelBase.__init__(self,configuration_dict, split_in_init)
        DataSummarizer.__init__(self)
 
    def __getitem__(self, index: int):
        import time
        """
        Retrieves an item by its index for model training/testing.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and its corresponding target tensor.
        """
        #process height data differently
        original_cn = self._cn
        if self.confi_datacube_transform['calculate_height']:
            height_channel = self.confi_datacube_transform['heigh_var_name']
            cntmp = [cn for cn in original_cn if cn != height_channel]
        else:
            cntmp = original_cn
            
        datatransformed,targetval = self.get_data(index, channel_names=cntmp)
        tmpdata = np.squeeze(datatransformed)
        quantiledata = self.summarise_into_quantiles(tmpdata, array_order = self._array_order, 
                                                     navalue=self.xrdata.attrs.get('nodata', 0),
            quantiles=self.confi['DATASETTRANSFORM']['quantiles'], channel_names =cntmp)
        
        if self.confi_datacube_transform['calculate_height']:
            imgred = self.confi['DATASETTRANSFORM']['height_percreduction']
            datatransformed_z,_ = self.get_data(index, channel_names=[height_channel], image_reduction=imgred, transform_options = ['raw'])
            self._tmpdata = datatransformed_z[0]
            quantiledata_z = self.summarise_into_quantiles(datatransformed_z[0],
                array_order = self._array_order, navalue=self.xrdata.attrs.get('nodata', 0),
                quantiles=self.confi['DATASETTRANSFORM']['height_quantile'],
                channel_names=[height_channel])
            quantiledata.update(quantiledata_z)
            #return original transformation

        return quantiledata, targetval
        

class MLDataReader(ClassificationMLData):
    
    def __init__(self, configuration_dict: dict, split_in_init: bool = True) -> None:
        super().__init__(configuration_dict, split_in_init)
        self._stratified_target = configuration_dict['DATASET'].get('stratified_target',False)
        
    @dask.delayed
    def _retrieve_single_data_dask(self, index):
        return self._retrieve_single_data(index)
        
    def _retrieve_single_data(self, index):
        dictfordf, targetval = self.__getitem__(index)
        input_df = from_quantiles_dict_to_df(dictfordf)#, idvalue=self._tr_data._file)
        return input_df, targetval
    
    def stack_data_as_table(self, cv:int = None, nkfolds:int = None, phase:str = 'train', nworkers:int = 0, ndatarepetions:int = 1, shuffle = False):
        """
        Prepare data for training by stacking inputs as a table. Optionally utilizes Dask for parallel processing.

        Parameters
        ----------
        cv : int, optional
            Index of the cross-validation split.
        nkfolds : int, optional
            Total number of cross-validation folds.
        phase : str, optional
            Specifies the phase of data to be loaded ('train', 'validation', 'test').
        nworkers : int, optional
            Number of worker threads to use if using Dask for parallel processing.
        
        Returns
        -------
        tuple
            Features as ndarray and targets as list.
        """
        # split data
        #if cv is not None:
        if ndatarepetions > 1:
            data_input, datatarget = [], []
            for i in range(ndatarepetions):
                di, dt = self.stack_data_as_table(cv = cv, nkfolds = nkfolds, phase = phase, nworkers = nworkers)
                data_input.append(di)
                datatarget.append(dt)
                
            data_target = datatarget[0]
            for i in range(1,len(datatarget)):
                data_target += datatarget[i]
                
            data_input = np.concatenate(data_input, axis = 0)
                
        else:
            self._split_data_in_traning_and_validation(cv = cv, nkfolds=nkfolds, phase = phase, stratified_target=self._stratified_target)   
            data_input, data_target = [], []
            if shuffle:
                rangedata = pd.Series(range(self.__len__())).sample(n = self.__len__(), random_state= 123).tolist()
            else:
                rangedata = range(self.__len__())
                
            if nworkers>0:
                #client = Client(n_workers=nworkers)
                results = [self._retrieve_single_data_dask(i) for i in rangedata]
                with ProgressBar():
                    results = dask.compute(*results)
                data_input, data_target = zip(*results) 
                #client.close()
            else:
                for i in tqdm(rangedata):
                    input_df, targetval = self._retrieve_single_data(i)
                    data_input.append(input_df)
                    data_target.append(targetval)
                    
            dfdata=  pd.concat(data_input)
            data_input = dfdata.drop(['id'], axis=1).values
        return data_input, data_target
        
        

class MASKRCNN_dataset(Dataset):
    
    def __init__(self,
                 cocodataset_reader,
                 augmentation_transform = None,
                 transform = None) -> None:

        self._imgreader = cocodataset_reader 
        self.augmentation_transform = augmentation_transform
        self.transform = transform
        
        
    def __len__(self):
        return self._imgreader.imgs_len

    def __getitem__(self, index):
    
        # get images rgb and segmented images
        self._imgreader.get_image_from_coco(index)
        # change order
        imgtr = np.einsum('HWC->CHW', self._imgreader._img.copy())
        imgmask = np.stack(self._imgreader._mask_imgid).copy()
        
        # apply transformation

        if self.augmentation_transform:
            
            imgtr, imgmask = self.augmentation_transform(imgtr,imgmask )
        
        ## bounding boxes
        bboxes = self._imgreader.bounding_boxes(imgmask)
        numobjs = len(bboxes)
        
        #bboxes = np.array([bboxes[0]]) if (bboxes.shape[0]>1) else bboxes
        # BNHW
        #imgmask = np.expand_dims(imgmask,axis =0) if len(imgmask.shape) == 3 else imgmask
        #print(imgmask.shape)
        
        #
        #imgtr = np.expand_dims(imgtr,axis =0) if len(imgtr.shape) == 3 else imgmask
        #print(imgtr.shape)                    
        ### transform to torch tensor
        if self.transform:
            imgtr, imgmask, bboxes = self.transform(imgtr, imgmask, bboxes)
            imgtr = imgtr.float() 
        else:
            imgtr = torch.from_numpy(imgtr).float()                
        
        # setting options
        labels = torch.ones((numobjs,), dtype=torch.int64)
        masks = torch.as_tensor(np.stack(imgmask), dtype=torch.uint8)
        iscrowd  = torch.zeros((numobjs,), dtype=torch.int64)
        image_id = torch.tensor([index])
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return imgtr, target
