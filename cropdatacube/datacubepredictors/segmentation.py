import math

import torch

import cv2
import numpy as np
import random
import os


from typing import List, Tuple, Optional, Callable

from ..utils.distances import euclidean_distance

from ..spatialdatacube.xr_functions import crop_xarray_using_mask, CustomXarray, from_dict_toxarray
from ..cropcv.mask_layer_fun import (getmidleheightcoordinates, 
                                     getmidlewidthcoordinates, get_boundingboxfromseg)

from ..cropcv.image_functions import read_image_as_numpy_array, resize_npimage
from ..cropcv.detection_plots import random_colors
from ..ml_utils.models.dl_architectures import maskrcnn_instance_segmentation_model

#
def get_heights_and_widths(maskcontours):

    p1,p2,p3,p4=maskcontours
    alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

    pheightu=getmidleheightcoordinates(p2,p3,alpharad)
    pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
    pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
    pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

    return pheightu, pheigthb, pwidthu, pwidthb



def get_height_width(grayimg, pixelsize: float = 1) -> Tuple[float, float]:
    """
    Calculate the height and width of an object in an image.

    Parameters:
    -----------
    grayimg : np.ndarray
        Grayscale input image.
    pixelsize : float, optional
        Size of a pixel. Defaults to 1.

    Returns:
    --------
    Tuple[float, float]
        Tuple containing the height and width of the object.
    """
    
    from ..cropcv.image_functions import find_contours
    
    ## find contours
    wrapped_box = find_contours(grayimg)

    ## get distances
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)
    
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)

    ## with this statement there is an assumption that the rice width is always lower than height
    larger = d1 if d1>d2 else d2
    shorter = d1 if d1<d2 else d2
    
    return larger* pixelsize, shorter*pixelsize


class ImageSegmentationBase():
    pass
    
          
class SegmentationDataCube(CustomXarray):
    """
    A class for handling data cube extending CustomXarray functionality,
    allowing operations for segmentation such as clipping based on mask layers and listing specific files.
    """
    
    @property
    def listcxfiles(self):
        """
        Retrieves a list of filenames ending with 'pickle' in the specified directory path.

        Returns
        -------
        Optional[List[str]]
            List of filenames ending with 'pickle', or None if path is not set.
        """
        
        if self.path is not None:
            assert os.path.exists(self.path) ## directory soes nmot exist
            files = [i for i in os.listdir(self.path) if i.endswith('pickle')]
        else:
            files = None
        return files
    
    
    def _clip_cubedata_image(self, min_threshold_mask:float = 0, padding: int = None):
        """
        Clips the cube data image based on the bounding box of the masking layer.

        Parameters
        ----------
        min_threshold_mask : float, optional
            Minimum threshold value for the mask. Defaults to 0.
        padding : Optional[int], optional
            Padding size. Defaults to None.

        Returns
        -------
        np.ndarray
            The clipped image as a numpy array.
        """
        ## clip the maksing layer 
        #maskbb = get_boundingboxfromseg(self._maskimg*255.)
        clipped_image = crop_xarray_using_mask(self._maskimg*255., self.xrdata, 
                                               buffer = padding,min_threshold_mask = min_threshold_mask)

        return clipped_image
        
    def _check_mask_values(self, mask_name, change_mask = False):
        """
        Checks and adjusts the mask image values based on predefined criteria.
        
        Adjusts the mask image by selecting an alternative mask if the sum is 0,
        squeezing the array if it has an unnecessary third dimension, and scaling
        the values if the maximum is not above a threshold.
        """
        if np.nansum(self._maskimg) != 0:
            if change_mask and len(self._msk_layers)>1:
                mask_name= [i for i in self._msk_layers if i != mask_name][0]
                self._maskimg = self.to_array(self._customdict,onlythesechannels = [mask_name])
            
        self._maskimg = self._maskimg.squeeze() if len(self._maskimg.shape) == 3 else self._maskimg
        self._maskimg = self._maskimg if np.max(self._maskimg) > 10 else self._maskimg*255.
        
    
    def mask_layer_names(self,
                         mask_suffix: str =  'mask',
                         ): 
        """
        Selects mask layer names from the cube data based on a suffix.

        Parameters
        ----------
        mask_suffix : str, optional
            Suffix to filter mask layers by. Defaults to 'mask'.

        Returns
        -------
        List[str]
            List of mask layer names.

        Raises
        ------
        ValueError
            If no mask layers are found.
        """
        # selcet mask name
        varnames = list(self.xrdata.keys())
        mask_layer_names = [i for i in varnames if i.startswith(mask_suffix)] if mask_suffix else None
        if not mask_layer_names:
            raise ValueError("There is no mask")
        
        self._msk_layers = mask_layer_names
        return mask_layer_names
    
    def clip_using_mask(self, 
                        mask_name: str = None, 
                        channels: List[str] = None,
                        padding: int = 0,
                        paddingincm: bool = False,
                        mask_data: bool = False,
                        mask_value: float = 0,
                        min_threshold_mask : float = 0):
        
        """
        Clip data using a specified mask.

        Parameters:
        -----------
        mask_name : str, optional
            Name of the mask. Defaults to None.
        channels : List[str], optional
            List of channels to clip. Defaults to None.
        padding : int, optional
            Padding size. Defaults to 0.
        paddingincm : bool, optional
            The padding size is in centimeters. Defaults to False
        mask_data : bool, optional
            Use the mask layer to mask the final datacube 
        min_threshold_mask : float, optional
            Minimum threshold value for the mask. Defaults to 0.
        Returns:
        --------
        np.ndarray
            Clipped image array.
        """
    
        channels = list(self.xrdata.keys()) if channels is None else channels
        # select mask name
        if self._msk_layers is not None:
            mask_name = random.choice(self._msk_layers) if mask_name is None else mask_name
        
        assert mask_name is not None
        
        # get data mask as array       
        self._maskimg = self.xrdata[mask_name].values# to_array(self._customdict, onlythesechannels = [mask_name])
        self._check_mask_values(mask_name)
        # padding in pixels
        if np.nansum(self._maskimg) != 0:
            padding =  int(padding/(self.xrdata.attrs['transform'][0]*100)) if paddingincm else padding
            
            ## clip the xarray            
            clipped_image = self._clip_cubedata_image(min_threshold_mask, padding) if np.nansum(self._maskimg) > 0 else self.xrdata.copy()

            if mask_data:
                clipped_image = clipped_image.where(clipped_image[mask_name]>min_threshold_mask,mask_value)
        else:
            clipped_image = None
            
        self._clippeddata = clipped_image
        
        return clipped_image
    
    
    def read_individual_data(self, file: str = None, path: str = None, 
                             dataformat: str = 'CHW') -> dict:
        """
        Read individual data from a file.

        Parameters:
        -----------
        file : str, optional
            Name of the file to read. Defaults to None.
        path : str, optional
            Path to the file directory. Defaults to None.
        dataformat : str, optional
            Data oder format. Defaults to 'CHW'.

        Returns:
        --------
        dict
        """
        if path is not None:
            file = os.path.basename(file)
        else:
            path = self.path
            file = [i for i in self.listcxfiles if i == file][0]
        
        self._arrayorder = dataformat
        customdict = self._read_data(path=path, 
                                   fn = file,
                                   suffix='pickle')
        
        self.xrdata  = from_dict_toxarray(customdict, dimsformat = dataformat)
        
        return customdict
            
        
    def __init__(self, path: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the CubeDataMetrics instance with the specified path and additional arguments.

        Parameters
        ----------
        path : Optional[str], optional
            The path to the directory containing data files, by default None.
        **kwargs : dict
            Additional keyword arguments passed to the CustomXarray parent class.
        """
        self.xrdata = None
        self._msk_layers = None
        self.path = path
        super().__init__(**kwargs)
        
        


class MASKRCNN_Detector(object):
    """
    A class for detecting objects using a pre-trained Mask R-CNN model.

    Attributes
    ----------
    predictions : list
        List to store predictions from the model.
    _frames_colors : list
        List to store colors for visualizing different instances.
    input_size : tuple
        The input size for the images.
    transform : callable
        Transform function to apply to the images.
    model : Any
        The pre-trained Mask R-CNN model.
    device : str
        The device to run the model on (e.g., "cpu", "cuda:0").

    Methods
    -------
    read_image(img_path: str) -> np.ndarray:
        Reads and processes an image from the file path.
    detect_layers(img_path: str = None, image_data: np.ndarray = None, threshold: float = 0.75, segment_threshold: int = 180) -> None:
        Detects objects in the image.
    _original_size() -> None:
        Resizes masks and bounding boxes to the original image size.
    _filter_byscore(threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Filters detections based on confidence score threshold.
    """
    
    def __init__(self, 
                 model, 
                 input_size: Tuple[int, int] = (512, 512),
                 transform: Optional[Callable] = None,
                 device: Optional[str] = None) -> None:
        
       
        """
        Initialize MaskRCNNDetector.

        Parameters
        ----------
        model : Any
            Pre-trained Mask R-CNN model for object detection.
        input_size : tuple of int, optional
            Input image size, by default (512, 512).
        transform : callable, optional
            Image transformation function, by default None.
        device : str, optional
            Device to use (e.g., "cpu", "cuda:0"), by default None.
        """
        
        self.predictions  = None
        self._frames_colors = None
        self.inputsize = input_size
        self.transform = transform
        self.model = model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def _resize_npimage(self, img):
        """
        Resize a NumPy image to the specified input size.

        Parameters
        ----------
        img : np.ndarray
            Input image as a NumPy array.

        Returns
        -------
        np.ndarray
            Resized image.
        """
        if self.inputsize is not None:
            img = resize_npimage(img, size= self.inputsize)
            self.keep_size = True
        else:
            self.keep_size = False
            
        return img
    
    def read_image(self, imgpath:str):
        """
        Read image from file.

        Parameters
        ----------
        img_path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            Image as a NumPy array.
        """
        assert os.path.exists(imgpath) ## path does not exist
        img = read_image_as_numpy_array(imgpath)
        self._origsize = img.shape[:2]
        return self._resize_npimage(img)
            
    def detect_layers(self,
        img_path: str = None,
        image_data: np.ndarray = None,
        threshold: float = 0.75,
        segment_threshold: int = 180):
        
        """
        Detect instance segmentation objects in the image.

        Parameters
        ----------
        img_path : str, optional
            Path to the image file, by default None.
        image_data : np.ndarray, optional
            Image as NumPy array with shape (H, W, C), by default None.
        threshold : float, optional
            Confidence threshold for detections, by default 0.75.
        segment_threshold : int, optional
            Threshold for segmenting masks, by default 180.

        Returns
        -------
        None
        """
        self._pathtoimg = img_path
        ## read image as H W C
        if img_path is not None:
            img = self.read_image(img_path)
        elif image_data is not None:
            img = image_data.copy()
            self._origsize = img.shape[:2]
            img = self._resize_npimage(img)
            
        
        
        if self.transform:
            imgtensor, _, _ = self.transform(img)
        else:
            imgtensor = torch.from_numpy(img/255).float()
                
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([imgtensor.to(self.device)])
        
                   
        self.predictions = prediction
        self.idimg = id
        self._imgastensor = imgtensor
        
        self.msks, self.bbs, self.labels = self._filter_byscore(threshold)
        
        for i in range(len(self.msks)):
            self.msks[i][self.msks[i]<segment_threshold] = 0
            

        self._frames_colors = random_colors(len(self.bbs))

        
        if self.keep_size:
            self._original_size()
            self._img= resize_npimage(img, size= self._origsize)          
        else:
            self._img = self._imgastensor.mul(255).permute(1, 2, 0).byte().numpy()
    
    def _original_size(self):
        """
        Resize masks and bounding boxes to original image size.

        Returns
        -------
        None
        """
        
        msksc = [0]* len(list(self.msks))
        bbsc = [0]* len(list(self.bbs))
        
        for i in range(len(self.msks)):
            msksc[i] = cv2.resize(self.msks[i], 
                                  [self._origsize[1],self._origsize[0]], 
                                  interpolation = cv2.INTER_AREA)  
            if len(bbsc)>0 and np.sum(msksc[i])>0:
                bbsc[i] = get_boundingboxfromseg(msksc[i])
            #else:
            #    bbsc[i] = []
        
        self.msks = np.array(msksc)
        self.bbs = np.array(bbsc)
        
    
    def _filter_byscore(self, threshold: float):
        """
        Filter detections based on confidence score threshold.

        Parameters
        ----------
        threshold : float
            Confidence threshold for detections.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Filtered masks, bounding boxes, and labels.
        """
        pred = self.predictions[0] 
        onlythesepos = np.where(
            pred['scores'].to('cpu').detach().numpy()>threshold)
        
        msks = pred['masks'].mul(255).byte().cpu().numpy()[onlythesepos, 0].squeeze()
        bbs = pred['boxes'].cpu().detach().numpy()[onlythesepos]
        labels = pred['labels'].cpu().detach().numpy()[onlythesepos]
        
        if msks.shape[0] == 0:
            msks = np.zeros(self.inputsize)
            
        if len(msks.shape)==2:
            msks = np.expand_dims(msks,0)
                
        return msks, bbs, labels
        
    

class DLInstanceModel:
    """
    Class for managing instance segmentation models.

    Attributes
    ----------
    model : torch.nn.Module
        Instance segmentation model.
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    device : str
        Device used for computation, defaults to 'cuda:0' if available, else 'cpu'.

    Methods
    -------
    __init__(weights=None, modeltype='instance_segmentation', lr=0.005, device=None)
        Initializes the DLInstanceModel instance.
    """
    
    
    def __init__(self, weights = None, modeltype = "instance_segmentation",
                 lr = 0.005, device = None, n_categories = 2) -> None:
        
        """
        Initializes the DLInstanceModel instance.

        Parameters
        ----------
        weights : str, optional
            Path to pre-trained model weights. Defaults to None.
        modeltype : str, optional
            Type of model to initialize. Defaults to "instance_segmentation".
        lr : float, optional
            Learning rate for optimizer. Defaults to 0.005.
        device : str, optional
            Device for computation. Defaults to None.

        Raises
        ------
        ValueError
            If the specified device is not available.
        FileNotFoundError
            If the model weights file does not exist.
        """
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if modeltype == "instance_segmentation":
            model = maskrcnn_instance_segmentation_model(n_categories).to(self.device)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=lr,
                                        momentum=0.9, weight_decay=0.0005)

        if weights:
            if not os.path.exists(weights):
                raise FileNotFoundError("Weights file not found.")
            
            if os.path.exists(weights):
                if self.device == "cpu":              
                    model_state, optimizer_state = torch.load(
                    weights, map_location=self.device)
                else:
                    model_state, optimizer_state = torch.load(
                    weights)
                
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                print("Weights loaded")
            
        self.model = model
        self.optimizer = optimizer
        