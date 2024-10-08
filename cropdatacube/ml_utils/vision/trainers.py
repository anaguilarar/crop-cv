import math
import sys
import time
import torch
import numpy as np

from torch.nn.functional import threshold, normalize

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .utils import reduce_dict, MetricLogger, SmoothedValue
from ..utils import warmup_lr_scheduler



def train_sam_one_epoch(model, optimizer,loss, data_loader, device, epoch, print_freq):
    model.to(device)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image["pixel_values"].to(device) for image in batch)
        boxesimg = list(box["input_boxes"].to(device) for box in batch)
        
        with torch.no_grad():
            image_embedding = model.image_encoder(
                torch.concat(images))
            
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=torch.concat(boxesimg),
                masks=None,
            )
        
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        maskupscaled_pred = model.postprocess_masks(low_res_masks, (256,256),(256,256))
        binary_mask = normalize(threshold(maskupscaled_pred, 0.0,0))

        truemask = torch.concat([torch.as_tensor(
                (np.expand_dims(image["ground_truth_mask"],axis = 0).astype(float)
                )) for image in batch])

        truemaskbin = torch.as_tensor(truemask>0, dtype= torch.float32).to(device)

        loss_red = loss(binary_mask, truemaskbin)

        loss_value = loss_red.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_red.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_red)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train_maskrcnn_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, reporter = None,
                             only_these_lossess = ['loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'],
                             grad_scaler = None):

    
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #with torch.cuda.amp.autocast():
        loss_dict = model(images, targets)
        
        if only_these_lossess is not None:
            losses = sum(loss for k, loss in loss_dict.items() if k in only_these_lossess)
        else:
            losses = sum(loss for k, loss in loss_dict.items())
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        optimizer.zero_grad()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if grad_scaler is None:
            losses.backward()
            optimizer.step()
        else:
            grad_scaler.scale(losses).backward()    
            grad_scaler.step(optimizer)
            grad_scaler.update()
        
        

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    if reporter is not None:
        epoch_losses = {k: metric_logger.meters[k].avg 
                        for k in list(metric_logger.meters.keys())[1:] if k in reporter._report_keys}
        epoch_losses['epoch'] = epoch
        
        reporter.update_report(epoch_losses)
        
    return reporter


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, reporter = None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
            
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, metric_logger

