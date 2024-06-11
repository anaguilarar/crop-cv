
from .image_functions import read_image_as_numpy_array
from ..utils.decorators import check_path
from ..utils.general import split_filename, FolderWithImages
import cv2
import os
import numpy as np

from typing import List

class ImageReader(object):
    """
    A class for managing image data.

    Attributes
    ----------
    path : str
        The path to the directory containing the image.
    img_name : str
        The name of the image file.
    """
    
    @property
    def images_names(self) -> dict:
        """
        Get image names.

        Returns
        -------
        dict
            A dictionary containing image names.
        """
        
        return {'raw': self.orig_imgname}

    @property
    def imgs_data(self) -> dict:
        """
        Get image data.

        Returns
        -------
        dict
            A dictionary containing image data.
        """
        imgdata = {'raw': self.img_data}
        augmented_imgs = self._augmented_images
        if augmented_imgs:
            imgdata.update(augmented_imgs)
        return imgdata
    
    
    @check_path
    def read_image(self, path: str, output_size: tuple = None) -> np.ndarray:
        """
        Read an image from the specified path.

        Parameters
        ----------
        path : str
            The path to the directory containing the image.
        fn : str
            The name of the image file.
        suffix : str, optional
            The suffix of the image file. Defaults to '.jpg'.
        output_size : tuple, optional
            The size of the output image. Defaults to None.

        Returns
        -------
        numpy.ndarray
            The image data as a NumPy array.
        """
        
        img = read_image_as_numpy_array(path, size= output_size)
        return img
    
    @staticmethod
    def _split_filename(filename: str) -> tuple:
        """
        Split a filename into directory path and file name.

        Parameters
        ----------
        filename : str
            The filename to split.

        Returns
        -------
        tuple
            A tuple containing the directory path and the file name.
        """
        
        dirname, filename = split_filename(filename)

        return dirname, filename
    
    def get_image(self, output_size: tuple = None, path: str = None) -> np.ndarray:
        """
        Get an image from the specified path.

        Parameters
        ----------
        output_size : tuple, optional
            The size of the output image. Defaults to None.
        path : str, optional
            The path to the directory containing the image. Defaults to None.

        Returns
        -------
        numpy.ndarray
            The image data as a NumPy array.
        """
        
        
        if path is not None:
            path, fn = self._split_filename(path)
        else:
            path, fn = self.path , self.img_name
        
        img = self.read_image(path = os.path.join(path, fn), output_size = output_size)

        return img
        
    
    def __init__(self, path: str = None, image_suffix = '.jpg') -> None:
        """
        Initialize the ImageData object.

        Parameters
        ----------
        path : str, optional
            The path to the directory containing the image. Defaults to None.
        """
        #FolderWithImages.__init__(self, path)
        self.img_name = None
        self.path = None
        if path is not None:
            self.path, self.img_name = self._split_filename(path)
