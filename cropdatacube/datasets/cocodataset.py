from ..cropcv.readers import ImageReader
from ..cropcv.image_functions import resize_npimage
from ..cropcv.mask_layer_fun import get_boundingboxfromseg
import numpy as np
import os
import shutil
import copy
from ..cropcv.bb_utils import xyxy_to_xywh

import datetime
import json

from typing import List, Dict

class CocoLabels():
    def __init__(self, path) -> None:
        foldername = os.path.dirname(path)
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            
        self.path = path
    
    def updating_coco_dataset(self,image_info, annotations):
        return update_cocotrainingdataset(self.path, image_info, annotations)
    
    @staticmethod
    def get_header(image_info, annotations, categorynames):
        
        ccoinput = cocodataset_dict_style(image_info, annotations, categorynames= categorynames)
        
        return ccoinput
    
    @staticmethod
    def transform_to_annotations(image_fn, image_id, img_height, img_width, bbs = None, masks = None, targets = None):
        annotations, image_info =  as_coco_dict_dataset(image_fn=image_fn,image_id=image_id,
                                                      bbs = bbs,labels=  targets, mask = masks, img_height=img_height, img_width=img_width)
    
        return image_info, annotations

    def export_dataset(self, cocodataset):
        with open(self.path , 'w', encoding='utf-8') as f:
            json.dump(cocodataset, f, indent=4)

def update_cocotrainingdataset(cocodatasetpath, newdata_images, newdata_anns):
    
    if os.path.exists(cocodatasetpath):
        with open(cocodatasetpath, 'r') as fn:
            previousdata = json.load(fn)
    else:
        previousdata = None
    
    if previousdata is not None:
        #newdatac = copy.deepcopy(newdata)

        #newdatac = copy.deepcopy(newdata)
        previousdatac = copy.deepcopy(previousdata)

        if isinstance(previousdatac["images"], dict):
            previousdatac["images"] = [previousdatac["images"]]
        
        if isinstance(previousdatac["annotations"], dict):
            previousdatac["annotations"] = [previousdatac["annotations"]]
            
        oldimageslist = copy.deepcopy(previousdatac["images"])
        oldannslist = copy.deepcopy(previousdatac["annotations"])
        
        if isinstance(newdata_images, dict):
            newimageslist = [copy.deepcopy(newdata_images)]
        else:
            newimageslist = copy.deepcopy(newdata_images)

        if isinstance(newdata_anns, dict):
            newannslist = [copy.deepcopy(newdata_anns)]
        else:
            newannslist = copy.deepcopy(newdata_anns)

        
        lastid = oldimageslist[len(oldimageslist)-1]['id']+1
        lastidanns = oldannslist[len(oldannslist)-1]['id']+1
        #print(oldannslist[0]['image_id'])
        for i, newimage in enumerate(newimageslist):
            previousid = newimage['id']
            newimage['id']  = lastid

            for j, newann in enumerate(newannslist):
                if isinstance(newann, list):
                    for k in range(len(newann)):
                        if newann[k]['image_id'] == previousid:
                            newann[k]['image_id'] = lastid
                            newann[k]['id'] = lastidanns
                            lastidanns+=1
                            
                else:

                    if newann['image_id'] == previousid:
                        newann['image_id'] = lastid
                        newann['id'] = lastidanns
                        lastidanns+=1
            lastid+=1
        
        #if len(newimageslist) == 1:
        #    newannslist = [newannslist]
        for newimage in newimageslist:
            previousdatac["images"].append(newimage)
        
        if isinstance(newannslist, list):
            for newann in newannslist:
                previousdatac["annotations"].append(newann)
        else:
            previousdatac["annotations"].append(newannslist)
                
    else:
        
        previousdatac = cocodataset_dict_style(newdata_images, newdata_anns)
        
        #previousdatac = copy.deepcopy(newdata)
        
    return previousdatac


def cocodataset_dict_style(imagesdict, 
                           annotationsdict: Dict,
                           categorynames: List = None,
                           exportpath: str = None, 
                           year: str = "2023"
                           ):
    
    cocodatasetstyle = {"info":{"year":year},
                    "licenses":[
                        {"id":1,"url":"https://creativecommons.org/licenses/by/4.0/",
                         "name":"CC BY 4.0"}]}
    categorynames = ['0'] if categorynames is None else categorynames
    
    categories = [{"id":i,"name":catname,
               "supercategory":"none"} for i, catname in enumerate(categorynames)]
    
    
    jsondataset = {"info":cocodatasetstyle,
        "categories":categories,
        "images":imagesdict,
        "annotations":annotationsdict}
    
    if exportpath:
        with open(exportpath, 'w', encoding='utf-8') as f:
            json.dump(jsondataset, f, indent=4)
        
    return jsondataset



def as_coco_dict_dataset(image_id, image_fn, img_height, img_width, bbs = None,mask = None, labels = None):
        
        dataanns = []
        for k in range(len(bbs)):
            xywh = xyxy_to_xywh(bbs[k])
            bb = [int(j) for j in bbs[k]]
            
            if mask is not None:
                binarymask = mask[k].copy()
                mask = decode_coco_masks(binarymask)
                
            category = labels[k] if labels is not None else 1
                
            dataanns.append(
            {"id":k,
            "image_id":image_id,
            "category_id":category,
            "bbox":bb, 
            "area":int(xywh[2]*xywh[3]),
            "segmentation":mask,
            "iscrowd":0
            })

        imgdata = {"id":image_id,
                   "license":1,
                   "file_name":image_fn,
                   "height":int(img_height),
                   "width":int(img_width),
                   "date_captured":datetime.datetime.now().strftime("%Y-%m-%d")
                   }

        return dataanns, imgdata

def decode_coco_masks(mask):
    import pycocotools.mask as mask_util
    if len(mask.shape) == 3:
        msk = mask[:,:,0]
    else:
        msk = mask
    
    binary_mask_fortran = np.asfortranarray(msk)
    rle = {'counts': [], 'size': list(binary_mask_fortran.shape)}
    msk = mask_util.encode(binary_mask_fortran)
    
    rle["counts"] = msk["counts"].decode()
    return rle


class CoCoImageryReader(ImageReader):
    
    _cocosuffix = '.json'
    _list_transforms = []
    _list_not_transform = ['clahe', 'hsv']
    
    @property
    def annotation_len(self):
        anns = self._annotation_cocodata.getAnnIds()
        return len(self.anns)
    
    @property
    def imgs_len(self):
        
        return self._cocoimgs_len()
    
            
    @property
    def target_data(self):
        #imgs_data = {'raw': self._mask_imgid}
        newdata = {i:img for i, img in enumerate(self._mask_imgid)}
        #if len(list(newdata.keys()))>0:
        #    imgs_data.update(newdata)

        return newdata

    @staticmethod
    def bounding_boxes(masklistimgs):
        bbs = []
        for i,mask in enumerate(masklistimgs):
            if np.sum(mask)>0:
                bbs.append(get_boundingboxfromseg(mask))
                
        boxes= np.array(bbs)
            
        return boxes
    
    @property
    def _mask_imgid(self):
        ### it must be a cocodataset
        assert self.cocodataset
            
        masks = np.array([np.array(self._annotation_cocodata.annToMask(ann) * ann["category_id"]
                        ) for ann in self.anns])
        
        if len(masks.shape) == 1:
            masks =  np.expand_dims(np.zeros(self._img.shape[:2]), axis = 0)    
        
        if len(masks.shape) == 2:
            masks =  np.expand_dims(masks, axis = 0)
            
        if masks.shape[1] != self.img_size[0] or masks.shape[2] != self.img_size[1]:
            masks = np.array([resize_npimage(mskid, size= self.img_size) for mskid in masks])
            #masks =  np.expand_dims(masks, axis = 0)
            
        return masks
    
    def copy_images_to(self, new_path):
        
        tmpbool = True
        i = 0
        while tmpbool:
            try:
                imgid = self._img_ids[i]
                img = self._annotation_cocodata.loadImgs(imgid)
                if os.path.exists(new_path):
                    src = os.path.join(self.input_path,img[0]['file_name'])
                    shutil.copy2(src, new_path)
                    print("Image {} copy to {}".format(src,new_path))
                
                i+=1
                
            except:
                tmpbool = False
        return i-1
    
    
    def _cocoimgs_len(self):
        imgles = len(self._img_ids)
        return imgles
    
    def get_image_from_coco(self, img_id):
        
        img_data = self._annotation_cocodata.loadImgs(self._img_ids[img_id])
        self.img_name = img_data[0]['file_name']
        assert len(img_data) == 1
        
        self.annotation_ids = self._annotation_cocodata.getAnnIds(
                    imgIds=img_data[0]['id'], 
                    catIds=1, 
                    iscrowd=None
                )
        
        self.anns = self._annotation_cocodata.loadAnns(self.annotation_ids)
        
        img = self.get_image(path = os.path.join(self.input_path, self.img_name), output_size=self.img_size)
        self._img = img
        return img
    
    
    def __init__(self, input_path, annotation_path, 
                    imgsize = None,
                    cocodataset = True) -> None:
        
        from pycocotools.coco import COCO
        
        
        self.input_path = input_path
        self.cocodataset = cocodataset
        self.img_size = imgsize
        ## must be a coco file
        if annotation_path.endswith('.json'):
            annotation = COCO(annotation_path)
        #assert isinstance(annotation, COCO)
        
        self._annotation_cocodata = copy.deepcopy(annotation)
        self._img_ids = list(self._annotation_cocodata.imgs.keys())
                
        super(CoCoImageryReader, self).__init__()
        
