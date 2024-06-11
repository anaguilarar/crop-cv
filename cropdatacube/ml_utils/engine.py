import torch

import logging

from .reporters import ReporterBase
import numpy as np
from collections import OrderedDict
from typing import List, Union, Dict
from types import SimpleNamespace 
from tqdm import tqdm
import warnings

import pandas as pd
from sklearn.metrics import f1_score

import copy

import pickle

from .utils import warmup_lr_scheduler, optimizer_to
from ..utils.decorators import check_output_fn
from ..phenotyping.utils import from_quantiles_dict_to_df


import os

class DLBaseEngine():
    """
    Base class for a deep learning training engine.

    This class defines the structure and required methods that every training engine must implement,
    including running iterations, saving models, and computing loss.

    Attributes
    ----------
    model : Torch.Model
        pyTorch base model
    iter : int
        Current iteration of the training loop.
    start_iter : int
        The iteration from which the training was started or resumed.

    Methods
    -------
    run_iter():
        A method to be implemented by subclasses to perform a single iteration of training.
    _save_loss(losses):
        Saves and logs losses from an iteration.
    save_model(path):
        Saves the model and optimizer states to a file.
    _write_metrics():
        Writes out current metric values using the configured reporter.
    compute_loss(pred, y):
        Computes the loss for a batch of predictions and ground truth labels.
    """
    
    def __init__(self, model) -> None:
        self.iter: int = 0
        self.start_iter: int = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model
    
    def run_iter(self):
        """
        Performs a single iteration of the training or evaluation process.
        Must be overridden by subclasses.
        
        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _save_loss(self, losses):
        """
        Saves and logs losses from a training iteration.

        Parameters
        ----------
        losses : dict or torch.Tensor
            The losses to log, which can be a single torch.Tensor or a dictionary of loss components.
        """
        #detectron2
        if isinstance(losses, dict):
            metrics_dict = {k: v.detach().cpu().item() for k, v in losses.items()}
            for k,v in metrics_dict.items():
               exec('self.'+k + '=v')
        else:
            self.loss = losses.detach().cpu().item() 
    
    def save_model(self, path):
        """
        Saves the model and optimizer states to the specified path.

        Parameters
        ----------
        path : str
            Path to the directory where the model and optimizer states will be saved.

        Raises
        ------
        AssertionError
            If the specified path does not exist.
        """
        #assert os.path.exists(path), "The specified path does not exist."
        
        torch.save(self.model.state_dict(),  path + '_model_params')
        torch.save(self.optimizer.state_dict(), path + '_optimizer_params')
        
        pathm = path + "_scaler_params"
        if self.grad_scaler:
            torch.save(self.grad_scaler.state_dict(), pathm)
    
    def load_weights(self, model_path: str, optimizer_path: str = None, scaler_path: str = None):
        """
        Loads the model, optimizer, and gradient scaler states from the specified paths.

        Parameters
        ----------
        model_path : str
            Path to the file from which the model state will be loaded.
        optimizer_path : str, optional
            Path to the file from which the optimizer state will be loaded.
        scaler_path : str, optional
            Path to the file from which the gradient scaler state will be loaded.

        Raises
        ------
        FileNotFoundError
            If any of the specified files does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path {model_path} does not exist.")
        model_state_dict = torch.load(model_path)
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        if optimizer_path and os.path.exists(optimizer_path):
            optimizer_state_dict = torch.load(optimizer_path)
            self.optimizer.load_state_dict(optimizer_state_dict)
        elif optimizer_path:
            raise FileNotFoundError(f"The optimizer path {optimizer_path} does not exist.")

        if scaler_path and os.path.exists(scaler_path):
            scaler_state_dict = torch.load(scaler_path)
            self.grad_scaler.load_state_dict(scaler_state_dict)
        elif scaler_path:
            raise FileNotFoundError(f"The scaler path {scaler_path} does not exist.")

        print("Model and other components (if specified) loaded successfully.")
            
    def _write_metrics(self):
        """
        Writes metrics using the reporter attribute if configured.
        """
        if self._reporter is None:
            return None
        
        values = {k: self.__getattribute__(k) for k in self._reporter._report_keys}
        self._reporter.update_report(values)
        
    def compute_loss(self, pred,y):
        """
        Computes the loss using the provided loss function.

        Parameters
        ----------
        pred : torch.Tensor
            Predictions made by the model.
        y : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        losses =  self.loss_fcn(pred,y)
        return losses
    



class DLTrainerModel(DLBaseEngine):
    """
    Training engine for deep learning models incorporating various functionalities such as
    training, validation, logging, and gradient scaling.

    Parameters:
    ----------
    model : nn.Module
        The neural network model to train.
    train_data_loader : DataLoader
        DataLoader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    validation_data_loader : DataLoader, optional
        DataLoader for validation data.
    reporter : Reporter, optional
        Tool to report metrics during training.
    grad_scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.
    loss_fcn : Callable, optional
        Loss function to be used during training.
    model_weight_path : str, optional
        Path to save the model weights.

    Attributes:
    ----------
    device : str
        Device to which the model and data are sent ('cuda' or 'cpu').
    """
    
    def __init__(self, model, train_data_loader,optimizer, validation_data_loader = None, 
                 reporter = None,
                 grad_scaler= None, loss_fcn = None, 
                 model_weight_path = None) -> None:
        
        super().__init__(model)

        self._tr_data_loader = train_data_loader
        self._val_data_loader = validation_data_loader
        self.optimizer = optimizer
        self._reporter = reporter
        self._multiclass = False
        self._weight_path = model_weight_path
        
        self.grad_scaler = grad_scaler

        self.loss_fcn = loss_fcn
        
        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = grad_scaler

        
        self.model = self.model.to(self.device)
        self.model.train()
        optimizer_to(self.optimizer,self.device)
        self.set_initial_params()
        self._set_initial_reporter_params()
    
    def set_initial_params(self):
        #iteration counting
        
        self.iter = 0
        self.eval_iter = 0
        # data loader
        self._data_loader_iter_obj = None
        self._data_loader_eval_iter_obj = None

        # training params
        self.lr_scheduler = None
        
        # iteration reporter
    def _set_initial_reporter_params(self):
        self._iter_tr_reporter = ReporterBase()
        self._iter_tr_reporter.set_reporter(['epoch', 'iter','loss'])
        self._iter_eval_reporter = ReporterBase()
        self._iter_eval_reporter.set_reporter(['epoch', 'eval_iter','eval_loss'])
        
    
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self._tr_data_loader)
        return self._data_loader_iter_obj
    
    @property
    def _data_loader_eval_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_eval_iter_obj is None and self._val_data_loader is not None:
            self._data_loader_eval_iter_obj = iter(self._val_data_loader)
        return self._data_loader_eval_iter_obj
        
    def compute_loss(self, pred,y):
        
        if pred.shape[1]>1:
            self._multiclass = True
            #pred = torch.max(pred, dim=1)[1]
            #pred = torch.unsqueeze(pred.type(torch.float32), dim = 1)
            #pred.requires_grad = True
            y = y.type(torch.int64)
            y = torch.squeeze(y)
        #if self._multiclass:
        #    losses.requires_grad = True
        losses =  self.loss_fcn(pred,y)
        
        return losses
    
    def fit(self, max_epochs: int, start_from: int = 0, 
            save_best: bool = False, 
            checkpoint_metric: str = 'eval_loss',
            best_value: float = 100.,
            suffix_model: str = None,
            lag_best: int = None,
            start_saving_from =None):
        """
        Run the training process for a specified number of epochs.

        Parameters:
        ----------
        max_epochs : int
            Total number of epochs to train the model.
        start_from : int, optional
            The starting epoch number, useful for resuming training. Default is 0.
        """
        start_saving_from = 0 if start_saving_from is None else start_saving_from
        self.epoch = int(start_from) if start_from else 0
        suffix_model = suffix_model if suffix_model else ""
        lag_best = max_epochs if lag_best is None else lag_best
        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {} to {}".format(self.epoch, max_epochs))
        
        pbar = tqdm(range(self.epoch, max_epochs),leave=True, desc="Overall Training Progress")
        bestloss = best_value
        lastbest_epoch = 0
        for _ in pbar:
            pbar.set_description("[Epoch %d]" % (self.epoch))
            
            self.train_one_epoch()
            
            if self._data_loader_eval_iter is not None:
                self.eval_one_epoch()

            epoch_metrics  = self.write_epoch_metrics(self.epoch)
            pbar.set_postfix(OrderedDict(epoch_metrics))
            self.epoch += 1
            self.set_initial_params() # Reset or update parameters if needed per epoch
            
            if bestloss> epoch_metrics[checkpoint_metric] and save_best:
                bestloss = epoch_metrics[checkpoint_metric]
                if self.epoch>start_saving_from:
                    outname = os.path.join(self._weight_path, self.model.model_name + suffix_model)
                    self.save_model(outname)
                    logging.info("The best model was saved at epoch: {} loss value: {:.4f}".format(self.epoch, bestloss))
                lastbest_epoch = self.epoch
            if lag_best < (self.epoch - lastbest_epoch):
                break
        if self._weight_path is not None:
            
            outname = os.path.join(self._weight_path, self.model.model_name+ '_last' + suffix_model)
            self.save_model(outname)
        
        
    def eval_one_epoch(self):
        """
        Evaluate the model on the validation dataset.
        """
        self.model.eval()
        max_iter = len(self._val_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='green', desc="Evaluating")
        for _ in pbar:
            try:
                #pbar.set_description("[iter %d]" % (self.eval_iter + 1))
                self.run_eval_iter()
                pbar.set_postfix(OrderedDict(loss=self.eval_loss))
                
            except Exception:
                warnings.warn("Exception during Evaluation:")
                
    
    def run_eval_iter(self):
        """
        Run a single evaluation iteration.
        """
        assert not self.model.training, "Model was changed to training mode!"
        
        x, y = next(self._data_loader_eval_iter)
        if isinstance(y[0], tuple):
            x = list(image.to(self.device) for image in x)
            y = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        else:
            y = y.to(self.device)
            x = x.to(self.device)
                    
        #with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = self.model(x)
            losses = self.compute_loss(output, y)
            
        #write losses losses
        self._save_loss(losses, evaluation =True)
        self._write_iter_metrics(evaluation= True)
        self.eval_iter +=1

    def train_one_epoch(self):
        """
        Conduct training over one epoch.
        """
        self.model.train()
        
        max_iter = len(self._tr_data_loader)
        pbar = tqdm(range(max_iter), leave=True, colour='blue', desc="Training")
        for _ in pbar:
        #    try:
                self.run_iter()
                pbar.set_postfix(OrderedDict(loss=self.loss))
        #    except Exception:
        #        warnings.warn("Exception during training:")
    def run_iter(self):
        """
        Run a single training iteration.
        """
        
        if self.iter == 0: self._init_lr_scheduler()
        
        assert self.model.training, "Model was changed to eval mode!"
        
        x, y = next(self._data_loader_iter)
        
        if isinstance(y, tuple):
            x = list(image.to(self.device) for image in x)
            y = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        else:
            y = y.to(self.device)
            x = x.to(self.device)
            
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if self.loss_fcn is not None:
                output = self.model(x)
                losses = self.compute_loss(output, y)
            else:
                loss_dict = self.model(x, y)
                losses = sum(loss for loss in loss_dict.values())
            
        #if self._multiclass:
        #    losses.backward()
        #    self.optimizer.step()
        #else:
        self.grad_scaler.scale(losses).backward()    
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
            
        #write losses losses
        self._save_loss(losses)
        self._write_iter_metrics()

        self.lr_scheduler.step()
        self.iter += 1
        
    def _save_loss(self, losses, evaluation=False):
        """
        Save loss value from the iteration.

        Parameters:
        ----------
        loss : float
            Loss value from the current batch.
        evaluation : bool, optional
            Flag to determine if the loss is from evaluation.
        """
        
        #detectron2
        
        if isinstance(losses, dict):
            metrics_dict = {k: v.detach().cpu().item() for k, v in losses.items()}
            for k,v in metrics_dict.items():
                if evaluation:
                    exec('self.eval_'+k + '=v')
                else:
                    exec('self.'+k + '=v')
        else:
            if evaluation:
                self.eval_loss = losses.detach().cpu().item() 
            else:
                self.loss = losses.detach().cpu().item() 
    
    def _write_iter_metrics(self, evaluation = False):
        if evaluation:
            values = {k: self.__getattribute__(k) for k in self._iter_eval_reporter._report_keys}
            self._iter_eval_reporter.update_report(values)
        else:
            values = {k: self.__getattribute__(k) for k in self._iter_tr_reporter._report_keys}
            self._iter_tr_reporter.update_report(values)

    
    def _calculate_metrics_fromreporter(self, reporter, epoch, dict_metrics = {}):
        iter_summary = reporter.summarise_by_groups(['epoch'])
        val =  iter_summary[str(epoch)]
        for j in val.keys():
            if j in self._reporter._report_keys:
                dict_metrics[j] = val[j]
                        
        return dict_metrics
    
    def write_epoch_metrics(self, epoch = 0):
        
        if self._reporter is None:
            return None
        values = {}
        values = self._calculate_metrics_fromreporter(self._iter_tr_reporter, epoch= epoch,dict_metrics = values)
            
        if self._data_loader_eval_iter is not None:
            values = self._calculate_metrics_fromreporter(self._iter_eval_reporter, epoch= epoch, dict_metrics = values)
        
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = os.path.join(self._weight_path, self._reporter.file_name), suffix = None)
        
        return values
        

                    
    def _init_lr_scheduler(self):
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(self._tr_data_loader) - 1)

        self.lr_scheduler = warmup_lr_scheduler(self.optimizer, 
                                           warmup_iters, warmup_factor)



class MLTrainerModel():
    """
    A model trainer for machine learning that utilizes data parallel processing
    and supports training with cross-validation.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        A scikit-learn compatible model that supports fit and predict methods.
    _tr_data : DataLoader
        A data loader that provides training and validation data.
    _reporter : ReporterBase
        An object to handle reporting of training metrics.
    _weight_path : str
        Path where the model weights and reports are saved.

    Methods
    -------
    train_model(xtr, ytr, xval, yval)
        Train the model on training data and evaluate on validation data.
    train_cv(nkfolds, nworkers, check_already_processed)
        Perform cross-validation training.
    """
    def __init__(self, model, 
                 reporter = None,
                 model_weight_path:str = None) -> None:
        """
        Initialize the MLTrainerModel with model, training data, reporter, and path for model weights.

        Parameters
        ----------
        model : sklearn.base.BaseEstimator
            The model to be trained.
        train_data : DataLoader
            The DataLoader providing access to the training and validation data.
        reporter : ReporterBase, optional
            The reporting object to log training process metrics.
        model_weight_path : str, optional
            Path to save the trained model weights.
        """
        super().__init__()
        self._rawmodel = copy.deepcopy(model)
        self._reporter = reporter
        self._weight_path = model_weight_path
        
        self.model = copy.deepcopy(model)
        
    

        # iteration reporter
    def train_model(self, xtr:np.ndarray,ytr:np.ndarray, xval:np.ndarray, yval:np.ndarray):
        """
        Train the model on training data and validate it on validation data.

        Parameters
        ----------
        xtr : ndarray
            Training features.
        ytr : ndarray
            Training targets.
        xval : ndarray
            Validation features.
        yval : ndarray
            Validation targets.

        Returns
        -------
        dict
            A dictionary containing the loss and other evaluation metrics.
        """
        self.model.fit(xtr, ytr)
        y_pred= self.model.predict(xval)
        losses = self.compute_loss(y_pred, yval)
        self._save_loss(losses)
        return losses
    

    
    def _check_already_done(self,nkfolds = None):
        #todo: check that those that have being trianed before
        listcv = []
        for cv in tqdm.tqdm(range(nkfolds)):
            if not os.path.join(self._weight_path, self._model_name+ '_cv_{}_of_.pickle'.format(cv, nkfolds)):
                listcv.append(cv)
            
        return listcv

    def train_cv(self, data_sets, nkfolds = None, nworkers = 0, init_cv = 0,check_already_processed = None, ndatarepetions=1, verbose = True):
        """
        Perform cross-validation training over a specified number of folds.

        Parameters
        ----------
        data_sets: list
            list of cross-validation dataset, the order is data[kfold][training, validation]
        nkfolds : int
            Number of folds to use for cross-validation.
        nworkers : int, optional
            Number of workers to use for parallel data retrieval and processing.
        check_already_processed : bool, optional
            Flag to check if some folds have already been processed and skip them.
        
        Raises
        ------
        ValueError
            If `nkfolds` is not specified or is set to a non-positive value.
        """
        if nkfolds is None or nkfolds <= 0:
            raise ValueError("Number of folds `nkfolds` must be a positive integer.")
        if verbose:
            loop = tqdm(range(init_cv,nkfolds))
        else:
            loop =range(init_cv,nkfolds)
        
        for cv in loop:
             ## train
            xtr, ytr = data_sets[cv][0]
            xval, yval = data_sets[cv][1]
        
            losses = self.train_model(xtr,ytr, xval, yval)
            if verbose:
                print(f"CV {cv}: Losses - {losses}")
            # Update the reporter with metrics
            self.cv = 0 if cv is None else cv
            self._write_metrics()
            self.save_model(os.path.join(self._weight_path, self._model_name+ '_cv_{}_of_{}.pickle'.format(cv, nkfolds)) )
            logging.info(f"saved in : {os.path.join(self._weight_path, self._model_name+ '_cv_{}_of_{}.pickle'.format(cv, nkfolds))}")
            self.model = copy.deepcopy(self._rawmodel)
    
            
        
    def _save_loss(self, losses: Dict, evaluation:bool=False):
        """
        Saves loss values from the evaluation.

        Parameters
        ----------
        losses : dict or float
            Loss values from the model evaluation.
        evaluation : bool, optional
            Indicates if the losses come from an evaluation phase.
        """
        
        #detectron2
        
        if isinstance(losses, dict):
            #metrics_dict = {k: v for k, v in losses.items()}
            for k,v in losses.items():
                if evaluation:
                    exec('self.eval_'+k + '=v')
                else:
                    exec('self.'+k + '=v')
        else:
            if evaluation:
                self.eval_loss = losses
            else:
                self.loss = losses
    
    def _write_metrics(self):
        values = {k: self.__getattribute__(k) for k in self._reporter._report_keys}
        self._reporter.update_report(values)
        self._reporter.save_reporter(path = os.path.join(self._weight_path, self._model_name + '_reporter.json'), suffix = None)
        
    @check_output_fn
    def save_model(self, path, fn = None):
        with open(os.path.join(path, fn), "wb") as file:
            pickle.dump(self.model, file)

    def compute_loss(self, pred,y):
        """
        Compute loss and other metrics for the model predictions against true labels.

        Parameters
        ----------
        pred : ndarray
            Predictions made by the model.
        y : ndarray
            True labels.

        Returns
        -------
        dict
            A dictionary containing computed metrics such as F1 score and accuracy.
        """
        y = np.expand_dims(np.array(y),1)
        pred = np.expand_dims(np.array(pred),1)
        model_accuracy = (1-((y != pred).sum()/y.shape[0]))
        model_f1score = f1_score(y, pred, average='weighted')
        
        losses =  {'f1score':model_f1score,'accuracy':model_accuracy}
        return losses
                    