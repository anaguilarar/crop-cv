import torch
import os

import numpy as np
from ..ml_utils.engine import DLBaseEngine
from tqdm import tqdm

from ..datasets.model_datasets import ClassificationTarget
import pandas as pd

from .base_processors import EvaluatorBase


import numpy as np
import os
import torch


def target_transform(target_values):
    
    newlabels = {str(val):str(i) for i, val in enumerate(np.unique(target_values))}
    newtrtarget = np.zeros(target_values.shape).astype(np.uint8)
    for val in np.unique(target_values):
        newtrtarget[target_values == val] = newlabels[str(val)]

    return newtrtarget



class ClassificationMLData(ClassificationTarget):
    def __init__(self, configuration_dict: dict)-> None : 

        """
        Initializes the data handler for ML models.

        Parameters
        ----------
        configuration_dict : dict
            Configuration dictionary specifying operational parameters.
        split_in_init : bool
            Split data in initialization
        """
        self.confi = configuration_dict
        self._data = None
        self._kfolds = self.confi['DATASPLIT']['kfolds']
        assert os.path.exists(self.confi['DATASET']['path']), 'the input path does not exist'
        ClassificationTarget.__init__(self,**self.confi)
        
        
    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_csv(self.confi['DATASET']['path'])
            
        return self._data
    
    def set_initial_params(self):
        #iteration counting
        self._idssubset = None
        self._valuesdata = None
    
    def select_features(self, features_list):
        
        return [i for i in self.data.columns if i in features_list]
        
    def _get_subsetdata(self, ids):
        subset = self.data.loc[[i in ids for i in self.data[self.confi['DATASET']['id_key']].values]] 
        
        target = subset[self.confi['DATASET']['target_key']].values
        colnames = self.select_features(self.confi['DATASET']['feature_names'].split('-'))
        input_data = subset[colnames]
        self.set_initial_params()
        return input_data, target
    
    def split_data_in_traning_and_validation(self,  cv = 1, phase = None):
        
        ids, _ = self.split_data(cv = cv, nkfolds = self._kfolds, phase = phase)
        input_data, target =self._get_subsetdata(ids)
        
        return input_data, target

    def split_data_in_kfolds(self):
        modeldataset = []

        for cv in range(self.confi['DATASPLIT']['kfolds']):
            trdata, trtarget = self.split_data_in_traning_and_validation(cv, phase= 'train')
            trtarget = target_transform(trtarget)
            valdata, valtarget = self.split_data_in_traning_and_validation(cv, phase = 'validation')
            valtarget = target_transform(valtarget)
            modeldataset.append([[trdata, trtarget],[valdata, valtarget]])
        return modeldataset
    
    
class DLEvaluatorModel(DLBaseEngine, EvaluatorBase):
    
    def __init__(self, model, device:str = None) -> None:
        
        DLBaseEngine.__init__(self,model)
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model.to(self.device)
        EvaluatorBase.__init__(self)
        
        
    def models_prediction(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def individual_prediction(self, nparray,  binary = True):

        data = nparray.copy()
        imgtensor = torch.from_numpy(np.squeeze(data)).float()
        imgtensor = torch.squeeze(imgtensor).float()
        ## CHW - > DCHW
        output = self.models_prediction(torch.unsqueeze(imgtensor, dim = 0))
        if binary:
            preds = torch.sigmoid(output)
            output = np.round(preds.to('cpu').detach().numpy())
        else:
            output = torch.max(output, dim=1)[1].to('cpu').detach().numpy()

        return output
    
    def predict(self, data, verbose=True, binary = True):
        if verbose:
            loop = tqdm(data)
        else:
            loop = data

        results = []
        #if == True

        for idx, (x, y) in enumerate(loop):
            output = self.models_prediction(x)
            y = y.to(self.device)
            # binary result
            if binary:
                preds = torch.sigmoid(output)
                output = np.round(preds.to('cpu').detach().numpy())
                results.append([y.to('cpu').detach().numpy().flatten(), output.flatten()])
            else:
                output = torch.max(output, dim=1)[1].to('cpu').detach().numpy()
                results.append([y.to('cpu').detach().numpy().flatten(), output])
        
        realval  =[]
        predval  =[]
        for i,j in results:
            for z, x in zip(i, j):
                realval.append(z.flatten())
                predval.append(x.flatten())
            
        return [realval, predval]


class DataCubeClassifierBase(EvaluatorBase):
    
    def  __init__(self, data_reader, multiple_layers = False) -> None:
        """_summary_
        -------------
        Parameters:
            evaluator_model (_type_): _description_
            data_reader (_type_): _description_
            multiple_layers (bool, optional): 
                if the datacube has multiple segmetnation layers. Defaults to False.
        """
        self.data_reader = data_reader
        
        self._multi_layer = multiple_layers
        super().__init__()
    
    
    def get_data(self, img_path, layer_names = None,  **kwargs):

        if layer_names:

            layernames = layer_names if isinstance(layer_names, list) else [layer_names]
            datatopredict = [self.data_reader.get_datacube(img_path,  
                                                channel_names = self.data_reader._cn,
                                                mask_name = layname, **kwargs) for layname in layernames]

        else:
            
            datatopredict  = self.data_reader.get_datacube(img_path,  
                                            channel_names = self.data_reader._cn,
                                            **kwargs)

            datatopredict = np.expand_dims(datatopredict, axis = 0)
            
        return np.array(datatopredict)
            
    def classify(self, img_path, classifier, binary_classification, **kwargs):
        
        datatopredict = self.get_data(img_path, **kwargs)
        output = [
            classifier.individual_prediction(datatopredict[0],  binary = binary_classification)
                  for i in range(datatopredict.shape[0])]
        
        return np.array(output)
    
    def ensemble_classification(self, data_path, classifier, list_model_paths, layer_names = None, binary_classification = True):
        ensemble_predictionresults = []
        for cv,model_fn  in enumerate(list_model_paths):
            #   models weight
            classifier.load_weights(model_path =model_fn, optimizer_path = None, scaler_path = None)
            # classification
            cat_pred = self.classify(data_path, classifier=classifier, binary_classification=binary_classification,
                                layer_names = layer_names)

            ensemble_predictionresults = ensemble_predictionresults + list(cat_pred)

        ensemble_predictionresults= np.array(ensemble_predictionresults)
        return ensemble_predictionresults 
