from typing import Tuple, Optional
from abc import ABCMeta, abstractmethod
import numpy as np
from ..cropcv.image_functions import clip_image_usingbb
from sklearn.metrics import f1_score

class EvaluatorBase():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def eval_loss(y_pred,y_obs):
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
        yobs = np.expand_dims(np.array(y_obs),1)
        ypred = np.expand_dims(np.array(y_pred),1)
        model_accuracy = (1-((yobs != ypred).sum()/yobs.shape[0]))
        model_f1score = f1_score(y_pred=ypred, y_true=yobs, average='weighted')
        
        losses =  {'f1score':model_f1score,'accuracy':model_accuracy}
        return losses
    
    
    

class CVDetector_base(metaclass=ABCMeta):
    
    @abstractmethod
    def read_image(self, imgpath:str):
        pass
    
    @abstractmethod
    def detect_objects(self, img_path: str = None, image: np.ndarray = None,
                    threshold: float = 0.75):
        pass

    @abstractmethod
    def _filter_byscore(self, threshold: float):
        pass 
    
    @abstractmethod
    def _original_size(self):
        pass
    
    @staticmethod
    def _clip_image(image: np.ndarray, bounding_box: Tuple[int, int, int, int],
                    bbtype: str = 'xyxy', padding: Optional[int] = None,
                    padding_with_zeros: bool = True) -> np.ndarray:
        """
        Clips an image to the specified bounding box, with optional padding.

        Parameters
        ----------
        image : np.ndarray
            The original image to be clipped.
        bounding_box : Tuple[int, int, int, int]
            The bounding box to clip the image to, specified as (x1, y1, x2, y2) for 'xyxy' type.
        bbtype : str, optional
            The type of bounding box ('xyxy' is currently the only supported format).
        padding : Optional[int], optional
            Optional padding to add to the bounding box dimensions. If None, no padding is added.
        padding_with_zeros : bool, optional
            If True, pads with zeros (background). If False, the behavior is undefined in this context, as 
            the function implementation for non-zero padding is not provided here.

        Returns
        -------
        np.ndarray
            The clipped (and potentially padded) image as a numpy array.

        Notes
        -----
        The `clip_image_usingbb` function is assumed to be defined elsewhere in the code. This function
        should handle the clipping and padding according to the specified parameters.
        """

        if bbtype == 'xyxy':
            x1,y1,x2,y2 = bounding_box
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
        imgclipped = clip_image_usingbb(image, [x1,y1,x2,y2], bbtype='xyxy', 
                                        padding=padding, paddingwithzeros = padding_with_zeros)
        
        
        return imgclipped