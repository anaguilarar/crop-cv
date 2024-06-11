import os
from .reporters import ReporterBase
from .engine import DLTrainerModel

import logging
import numpy as np
import yaml
import torch


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)



def target_transform(target_values: np.ndarray) -> np.ndarray:
    """
    Transform target values to a new set of integer labels.

    Parameters
    ----------
    target_values : np.ndarray
        Original target values.

    Returns
    -------
    np.ndarray
        Transformed target values as integers.
    """
    newlabels = {str(val):str(i) for i, val in enumerate(np.unique(target_values))}
    newtrtarget = np.zeros(target_values.shape).astype(np.uint8)
    for val in np.unique(target_values):
        newtrtarget[target_values == val] = newlabels[str(val)]

    return newtrtarget



def check_model_folders(path, run_suffix = '_run_'):
    if os.path.exists(path):
        runnumber = path[path.index(run_suffix)+len(run_suffix):len(path)]
        path = path[0:path.index(run_suffix)] + run_suffix + '{}'.format(int(runnumber)+1)
        path = check_model_folders(path)
    else:
        os.mkdir(path)
        
    return path

def create_run_folder_from_config(config, run_suffix = "_run_0"):
    pathbase = os.path.dirname(config['MODEL']['weight_path']) if config['MODEL']['weight_path'].endswith('/') else config['MODEL']['weight_path']
    outputpath = pathbase + '_' + ''.join(
        [i[:2] for i in config['DATASET']['feature_names'].split('-')])
    outputpath = outputpath + run_suffix
    newpath = check_model_folders(outputpath)
    return newpath


def setup_reporter(config, counter = 'epoch'):
        ## set reporter
    reporter = ReporterBase()
    if config['start_from_scratch']:
        reporter.set_reporter(config['reporter_keys'])
        init_epoch = 0
        bestlossvalue = float('inf')
    else:
        pathtoreport = os.path.join(config['reporter_path'],config['file_name'])
        reporter.load_reporter(pathtoreport)
        init_epoch = reporter.report[counter][-1] + 1
        evallosvalues = reporter.report['eval_loss']
        bestlossvalue = np.min(evallosvalues) + np.std(evallosvalues)
        
        logging.info("The reporter was uploaded best loss validation value {:.4f}".format(bestlossvalue))

    return reporter, init_epoch, bestlossvalue



def set_configuration(config_path, phase= 'train', cv = -1):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} path does not exists")
    
    with open(config_path, 'r') as fn:
        opt =  yaml.load(fn, Loader=yaml.SafeLoader)

    if phase == 'train':
        opt['DATASET']['phase'] = 'train'
        opt['DATASET']['path'] = opt['DATASET']['train_file_path']
        opt['DATASET']['only_these_tps'] =opt['DATASET'].get('validation_time_points', None)
    else:
        opt['DATASET']['phase'] = 'validation'
        opt['DATASET']['path'] = opt['DATASET']['validation_file_path']
        opt['DATASET']['only_these_tps'] = opt['DATASET'].get('validation_time_points', None)
        
    opt['DATASPLIT']['cv'] = cv
    return opt



def select_columns(df, colstoselect, additionalfilt = 'gr_'):
    colsbol = [i.startswith(colstoselect[0]
    ) and additionalfilt not in i for i in df.columns]

    for cols in range(1, len(colstoselect)):
        colsbol = np.array(colsbol) | np.array([i.startswith(
            colstoselect[cols]) and additionalfilt not in i for i in df.columns])

    return df[list(compress(df.columns,  colsbol))]



def setup_model(config):
    from .models.dl_architectures import HybridUNetClassifier, HybridCNNClassifier, ClassificationCNNtransformer, pre_trainedmodel
    nchannels = config['n_channels']
    noutputs = config.get('n_classess', 1)
    print(f'n outputs : {noutputs}')
    if config['backbone'] == 'Unet':
        model = HybridUNetClassifier(in_channels =nchannels, 
                                    out_channels=config['backbone_out_channels'],
                                    features=config['backbone_features'], 
                                    classification_model= config['classifier_name'],
                                    nlastlayer=noutputs)

    elif config['backbone'] == 'CNN' and config['classifier_name'] == 'transformer':
        model = ClassificationCNNtransformer(in_channels =nchannels, 
                                    features=config['backbone_features'], 
                                    classification_model= config['classifier_name'],
                                    strides= config['strides'],
                                    blockcnn_dropval= config['blockcnn_drop'],
                                    nlastlayer=noutputs)

    elif config['backbone'] == 'CNN':
        model = HybridCNNClassifier(in_channels =nchannels, 
                                    features=config['backbone_features'], 
                                    classification_model= config['classifier_name'],
                                    strides= config['strides'],
                                    n_lastlayer=noutputs)
        
    elif config['backbone'] == '':
        model = pre_trainedmodel(config['classifier_name'], nchannels, noutputs)
        model.model_name = config['classifier_name']
        
    logging.info(f"The {model.model_name} model was intitialized")


    return model



def run_one_model(config: dict, training_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader) -> None:
    """
    Run a single model training process.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model, training, and reporting settings.
    training_data : torch.utils.data.DataLoader
        DataLoader for the training data.
    validation_data : torch.utils.data.DataLoader
        DataLoader for the validation data.

    Returns
    -------
    None
    """
    import torch.nn as nn
    
    cv = config['DATASPLIT'].get('cv', None)
    kfold = config['DATASPLIT'].get('kfolds', None)
    cvlabel = '_cv_{}_of_{}'.format(cv,kfold) #if cv else ''
        
    # model's configuration
    config['MODEL']['n_channels'] = len(config['DATASET']['feature_names'].split('-'))
    model = setup_model(config['MODEL'])
    optimizer = torch.optim.SGD(model.parameters(), lr=10**-2, momentum=0.9)
    
    # Loss function configuration
    lossfunction = config['MODEL'].get('loss_function',None)
    if lossfunction == 'cross_entropy':
        print('cross entropy')
        lossfun = torch.nn.functional.cross_entropy
    else:
        lossfun = nn.BCEWithLogitsLoss()
    
    # Reporter's configuration
    config['REPORTER']['file_name'] = model.model_name+'_reporter.json'
    config['REPORTER']['reporter_path'] = config['MODEL']['weight_path']
    reporter, init_epoch, bestlossvalue = setup_reporter(config['REPORTER'])
    config['TRAINING']['best_loss_value'] = bestlossvalue
    reporter.file_name = model.model_name + '{}_reporter.json'.format(cvlabel)
    dltrainer = DLTrainerModel(model = model, 
                            train_data_loader=training_data, 
                            validation_data_loader= validation_data, 
                            optimizer=optimizer, 
                            model_weight_path = config['MODEL']['weight_path'],
                            reporter=reporter, 
                            loss_fcn=lossfun)
    
    if config['TRAINING']['start_from_scratch']:
        mpath = os.path.join(dltrainer._weight_path,model.model_name + '_model_params')
        opath = os.path.join(dltrainer._weight_path,model.model_name + '_optimizer_params')
        scpath = os.path.join(dltrainer._weight_path,model.model_name + '_scaler_params')
        dltrainer.load_weights(model_path=mpath,optimizer_path=opath, scaler_path=scpath)
        logging.info(f"The {dltrainer.model.model_name} model was uploaded from {mpath}")
    else:
        logging.info(f"The {dltrainer.model.model_name} model is gonna start from {init_epoch}")

    with open(os.path.join(config['MODEL']['weight_path'], model.model_name + '_configuration.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
    logging.info(f"""Training: {dltrainer.model.model_name}
                 Model will be saved in: {config['MODEL']['weight_path']}
                 n Channels: {config['MODEL']['n_channels']}
                 reporter: {reporter.file_name} """)
    
    # trainining start
    dltrainer.fit(
        start_from= int(init_epoch),
        max_epochs= int(config['TRAINING']['max_epochs'] + init_epoch), 
        save_best=config['TRAINING']['save_best'],
        best_value= config['TRAINING']['best_loss_value'],
        checkpoint_metric = config['TRAINING']['checkpoint_metric'],
        suffix_model='{}.pickle'.format(cvlabel),
        lag_best=config['TRAINING']['stopat'],
        start_saving_from = config['TRAINING']['startsavingfrom'])