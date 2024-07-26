import os

import logging
import numpy as np
import yaml
import torch


from torchvision.ops.boxes import box_area


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



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


