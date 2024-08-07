import torch
import tqdm
import numpy as np
import itertools
from sklearn.metrics import r2_score
import os

from .dl_architectures import *


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], :img.shape[2]] = False
        else:
            raise ValueError('not supported')
        return cls(tensor, mask)
    
    def filter_batches(self,bs_to_keep):
        self.tensors = self.tensors[bs_to_keep]
        self.mask = self.mask[bs_to_keep]
        
        
    def __repr__(self):
        return repr(self.tensors)
    


def check_already_processed(reporter, currentconf):
    datafeatunique =[]
    
    for i in range(len(reporter.reporter['modelname'])):
        attr = [reporter.reporter[attr][i] for attr in ['modelname', 'features_names','cv']]
        if attr not in datafeatunique:
            datafeatunique.append(attr)
    
    currentconfig = currentconf
    
    return currentconfig in datafeatunique


def load_model_weights(model, optimizer, modelname, path, uploadlast = False):
    if path is not None:
        suffixmodel = "_last" if uploadlast else "_checkpoint"
        #suffixmodel = suffixmodel + ".zip" if modelname == "transformer" else suffixmodel
        weightspath = os.path.join(path,
                                        modelname+suffixmodel)
        print(weightspath)
        if os.path.exists(weightspath):
            print("loading weigths from: " + weightspath)
            model_state, optimizer_state = torch.load(weightspath)
                
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            print('***** model uploaded')
                
        elif not os.path.exists(path):
            os.mkdir(path)
    
    return model, optimizer


def model_setting(modelname, config, checkpointdir = None, uploadlast = False):
    if modelname == "cnn3d":
        model = CNN3DRegression(in_channels=config['n_channels'],
                                        features = config['features'], 
                                        out_channels=1,
                                        in_times=config['n_timepoints'],
                                        widthimg=config['width'],
                                        #fc = config['fc'], 
                                        #use_global = config['use_global'],
                                        fc_dropval=config['fc_dropval'])
    
    if modelname == "cnn_transformer":
        model = ClassificationCNNtransformer(in_channels=config['n_channels'],
                                  features = config['features'],
                                  blockcnn_dropval=config['fc_dropval'],
                                  strides = config['strides'])
        
    
    if modelname == "cnn3d_transformer":
        model = RegressionDLModel(in_channels=config['n_channels'],
                                  in_times = config['n_timepoints'],
                                  features = config['features'],
                                  block3dconv_dropval=config['fc_dropval'],
                                  strides = config['strides'])

    if modelname == "cnn3d_resnet":
        model = RegressionResNetModel(in_channels=config['n_channels'],
                                  in_times = config['n_timepoints'],
                                  features = config['features'],
                                  block3dconv_dropval=config['fc_dropval'],
                                  strides = config['strides'])
    
    if modelname == "cnn3d_transformer_classification":
        model = Classification3DCNNtransformer(in_channels=config['n_channels'],
                                  in_times = config['n_timepoints'],
                                  features = config['features'],
                                  block3dconv_dropval=config['fc_dropval'],
                                  strides = config['strides'])
        
        
    if modelname == "cnn3d_efficient_classification":
        model = Classification3DCNNEfficientNet(in_channels=config['n_channels'],
                                  in_times = config['n_timepoints'],
                                  features = config['features'],
                                  block3dconv_dropval=config['fc_dropval'],
                                  strides = config['strides'])
    if modelname == "transformer":
        model = TransformerMLTC(in_channels=config['n_channels'],
                                  nlastlayer =1)
        
        

    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    
    model, optimizer = load_model_weights(model, optimizer, modelname = modelname, path = checkpointdir, uploadlast=uploadlast)

    return model, optimizer



def eval_classification_fn(model, 
            dataloader, 
             loss_fn,
             epoch,
             progressbar = True):
    
    if progressbar:
        loop = tqdm.tqdm(dataloader, desc = f'Epoch {epoch}', leave=False)
    else:
        loop = dataloader      
    
    loop = tqdm.tqdm(dataloader, desc = f'Epoch {epoch}', leave=False)
    g_loss = 0
    running_corrects = 0
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    totalelements = 0
    model.eval()
    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(x)
            #loss = loss_fn(output, y)
        
        loss = loss_fn(output, y)   
        preds = torch.sigmoid(output)
        predround = np.round(preds.to('cpu').detach().numpy())
        running_corrects += (predround.flatten() != y.to('cpu').detach().numpy().flatten()).sum()
        totalelements += x.shape[0]
        g_loss += loss.item()
    
    return g_loss/dataloader.__len__(), (1- (float(running_corrects)/(totalelements)))


## reporter utils

def update_reporter(reporter, toreport, path = None, reportervals = None, dl = True, fn = 'ml_reporter.json'):

    
    if reportervals is not None:
        reportervalsc = reportervals.copy()
        reportervalsc.update(toreport)
        
    else:
        reportervalsc= toreport
        
    reporter.update_reporter(reportervalsc)
    if path is not None:
        if dl:
            reporter.save_reporter(os.path.join(path,fn))
        
        else:
            reporter.save_reporter(os.path.join(path,fn))
                
    
    return reporter

