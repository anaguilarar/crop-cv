import torch.nn as nn
import torch
from typing import List,Dict

#modified from https://github.com/ziqi-jin/finetune-anything/blob/main/losses

LOSSES = {'ce': torch.nn.functional.cross_entropy, 
          'bce':nn.BCEWithLogitsLoss,
          'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
          'mse': nn.MSELoss}


def set_loss_function(loss_names: Dict):
    
    loss_dict = {}
    for loss_n in loss_names:
        assert loss_n in LOSSES, print('{name} is not supported, please implement it first.'.format(name=loss_n))
        if loss_n[loss_n]['params'] is not None:
            loss_dict[loss_n] = LOSSES[loss_n](**loss_names[loss_n]['params'])
        else:
            loss_dict[loss_n] = LOSSES[loss_n]()
            
    return loss_dict