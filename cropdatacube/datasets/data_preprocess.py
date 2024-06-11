
from torch.nn import functional as F
import torchvision.transforms as transforms

from .utils import scale_data
import numpy as np
import torch

from typing import List


def get_padding_dims(heigth, width):
    max_dim = max(width, heigth)
    pad_w = (max_dim - width) // 2
    pad_h = (max_dim - heigth) // 2
    padding = (pad_w, pad_h, max_dim - width - pad_w, max_dim - heigth - pad_h)
    
    return padding


## instance segmentation
class PreProcess_InstaData():
    """
    A class for preprocessing instance segmentation data.

    Attributes:
        _mean_scaler_values (list): The mean scaler values for standardization.
        _std_scaler_values (list): The standard deviation scaler values for standardization.
    """
        
    def __init__(self, mean_scaler_values = None, std_scaler_values = None, new_size = None) -> None:
        """
        Initialize the PreProcess_InstaData object.

        Args:
            mean_scaler_values (list, optional): The mean scaler values for standardization. Defaults to None.
            std_scaler_values (list, optional): The standard deviation scaler values for standardization. Defaults to None.
        """

        self._mean_scaler_values = mean_scaler_values
        self._std_scaler_values = std_scaler_values
        self._new_size = new_size
        
    def __call__(self, image, masks = None, bboxes = None):
        """
        Perform preprocessing on the input image, masks, and bounding boxes.

        Args:
            image (numpy.ndarray): The input image as a NumPy array with shape (C, H, W) if the image shape is (H, W, C) the image will be transformed.
            masks (list, optional): The list of masks as NumPy arrays. Defaults to None.
            bboxes (list, optional): The list of bounding boxes as tuples. Defaults to None.

        Returns:
            tuple: A tuple containing the preprocessed image as a PyTorch tensor, preprocessed masks, and preprocessed bounding boxes.
        """
        if image.shape[2] < 10:
            image = np.einsum('HWC->CHW', image)
            
        _,h,w = image.shape

        if np.max(image) > 100:
            image = image.astype(np.float32)/255.

        if self._mean_scaler_values is not None:

            image = self.apply_standardization(image, 
                                       self._mean_scaler_values,
                                       self._std_scaler_values)


        #image = self.pad_image(image)
        
        #masks = [self.pad_image(mask) for mask in masks]
        if bboxes is not None:
            #bboxes = self.pad_boxes(h,w,bboxes)
            bboxes = np.array(bboxes)
        
        #image = self.to_tensor(image)
        image = torch.from_numpy(image).float()
        
        return image, masks, bboxes
    
   
    @staticmethod
    def pad_boxes(h,w,bboxes):
        """
        Pad bounding boxes to match the padded image dimensions.

        Args:
            h (int): The height of the image.
            w (int): The width of the image.
            bboxes (list): The list of bounding boxes as tuples.

        Returns:
            list: The list of padded bounding boxes.
        """
        pad_w, pad_h, _ , _ = get_padding_dims(h,w)
        ## bboxes : B, corners
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]
        
        return bboxes
    
    @staticmethod
    def pad_image(image):
        """
        Pad the input image to make it square.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: The padded image as a NumPy array.
        """
        ## resize image to the longest size
        if len(image.shape) == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape

        padding = get_padding_dims(h, w)
        
        image = transforms.Pad(padding)(image)
        
        return image
    
    @staticmethod
    def apply_standardization(image: np.array, meanval: List[float] = None, stdval: List[float] = None) -> np.array:
        """
        Apply standardization to the input image.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.
            meanval (list, optional): The mean scaler values for standardization. Defaults to None.
            stdval (list, optional): The standard deviation scaler values for standardization. Defaults to None.
        """

        if meanval is not None:
            
            scaleddata = scale_data(
                image, {'mean':meanval, 
                           'std':stdval})
        else:
            scaleddata = scale_data(image, None)
            
        return scaleddata
        
        
