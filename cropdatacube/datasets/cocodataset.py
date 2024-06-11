from ..cropcv.readers import ImageReader
from ..cropcv.image_functions import resize_npimage
from ..cropcv.mask_layer_fun import get_boundingboxfromseg
import numpy as np
import os
import shutil
import copy

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
        
