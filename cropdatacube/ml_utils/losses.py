import torch.nn as nn
import torch

import torch.nn.functional as F

from typing import List,Dict

from .utils import (
    box_cxcywh_to_xyxy, 
    generalized_box_iou
)

def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


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



def loss_labels(target, predictions, indices, num_classes, empty_weight):
    

    src_logits = predictions['pred_logits']

    idx = get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(target, indices)])


    target_classes = torch.full(src_logits.shape[:2], num_classes,
                            dtype=torch.int64, device=next(iter(predictions.values())).device)

        
    target_classes[idx] = target_classes_o
            
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2).float(), target_classes, empty_weight)
    losses = {'loss_ce': loss_ce}
    
    return losses


def loss_boxes(target, predictions, indices, num_boxes):

    # Compute all the requested losses
    idx =  get_src_permutation_idx(indices)
    src_boxes = predictions['pred_boxes'][idx]

    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(target, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    
    return losses

    
class DETR_Losses(nn.Module):
    
    #@property
    #def _loss_functions()
    
    def __init__(self, matcher, num_classes, eos_coef: float = 0.1, weight_dict = None, device = None):
        super().__init__()
        
        self.matcher = matcher
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.L1Loss()
        self.eos_coef = eos_coef
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1, device=device)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight
        self.device = device
        #self.register_buffer('empty_weight', empty_weight)
    
    def forward(self, outputs, targets):
        
        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        #num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()
        
        lossbox = loss_boxes(targets, outputs,  indices, num_boxes)

        losslabel = loss_labels( targets, outputs,indices, self.num_classes, self.empty_weight)
        
        losses = {}
        losses.update(lossbox)
        losses.update(losslabel)
        
        return losses
    