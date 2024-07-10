from typing import List, Optional, Union, Any
from .utils import perform, perform_kwargs, summarise_trasstring
from ..datasets.utils import standard_scale, minmax_scale
from .image_functions import (image_rotation,image_zoom,randomly_displace,
                              clahe_img, read_image_as_numpy_array,
                              image_flip,shift_hsv,diff_guassian_img,
                              illumination_shift, from_array_2_jpg,
                              shear_image,cv2_array_type,perspective_transform)


from .bb_utils import percentage_to_bb

from tqdm import tqdm


import numpy as np
import cv2
import copy

import random
from PIL import Image


class ImageAugmentation(object):
    """
    A class for performing image augmentation through various transformations like rotation, zoom, etc.
    Allows randomization of parameters for each transformation and maintains a history of applied transformations.
    """

    
    def __init__(self, img: Union[str, np.ndarray] = None, 
                 random_parameters: Optional[dict] = None, 
                 multitr_chain: Optional[List[str]] = None) -> None:
        """
        Initializes the ImageAugmentation class with an image and optional transformation parameters.

        Parameters:
        ----------
        img : str or ndarray
            The path to an image file or the image data itself.
        random_parameters : dict, optional
            Custom parameters for random transformations.
        multitr_chain : list, optional
            A predefined chain of transformations to apply.
        """
        
        self._transformparameters = {}
        self._new_images = {}
        self.tr_paramaters = {}
        
        self.img_data = None
        if img is not None:
            self.img_data = cv2.imread(img) if isinstance(img, str) else copy.deepcopy(img)

        self._init_random_parameters = random_parameters
        self._multitr_chain = multitr_chain
        
    
    @property
    def available_transforms(self):
        return list(self._run_default_transforms.keys())
    
    @property
    def _run_default_transforms(self):
        
        return  {
                'rotation': self.rotate_image,
                'zoom': self.expand_image,
                'clahe': self.clahe,
                'shift': self.shift_ndimage,
                'multitr': self.multi_transform,
                'flip': self.flip_image,
                'hsv': self.hsv,
                'shear': self.shear_image,
                'perspective': self.perspective_image,
                #'gaussian': self.diff_gaussian_image,
                'illumination':self.change_illumination    
            }
        
    @property    
    def params_kwargs(self):
        return {
            'rotation': 'angle',
            'zoom': 'zoom_factor',
            'clahe': 'thr_constrast',
            'shear': ['shear_x','shear_y'],
            'shift': ['xshift','yshift'],
            'flip': 'flipcode',
            'gaussian': 'high_sigma',
            'hsv': 'hsvparams',
            'illumination': 'illuminationparams'
            }

    @property
    def _augmented_images(self):
        return self._new_images

    @property
    def _random_parameters(self):
        params = None
        default_params = {
                'rotation': random.randint(10,350),
                'zoom': random.choice([0.1,0.2,0.3, 0.4,0.5]),
                'clahe': random.randint(5,30),
                'shear': [random.choice(np.linspace(5,40,8)/100),
                          random.choice(np.linspace(5,40,8)/100)],
                'perspective': [random.choice(np.linspace(-10,10,10)/10000) ,
                                random.choice(np.linspace(-10,10,10)/10000) ],
                
                'shift': random.randint(5, 20),
                'flip': random.choice([-1,0,1]),
                'gaussian': random.choice([20,30,40, 50]),
                'hsv': [random.choice(list(range(-10,30,5))),
                        random.choice(list(range(-10,20,5))), 
                        random.choice(list(range(-10,20,5)))],
                'illumination':random.choice(list(range(-50,50,5)))
            }
        
        if self._init_random_parameters is None:
            params = default_params

        else:
            params = self._init_random_parameters
            assert isinstance(params, dict)
            for i in list(self._run_default_transforms.keys()):
                if i not in list(params.keys()) and i != 'multitr':
                    params[i] = default_params[i]        
            
        return params    
    
    def _select_random_transforms(self, n_chains = 4):
        """
        Randomly selects a set of transformations for multi-transform.

        Returns:
        -------
        list
            A list of randomly selected transformation names.
        """
        chain_transform = []
        while len(chain_transform) < n_chains:
            trname = random.choice(list(self._run_default_transforms.keys()))
            if trname != 'multitr' and trname not in chain_transform:
                chain_transform.append(trname)
                
        return chain_transform

    def updated_paramaters(self, tr_type):
        """
        this function updates the tranformation dictionary information
        
        Parameters:
        ----------
        tr_type : str, optional
            transformation name
        """
        self.tr_paramaters.update({tr_type : self._transformparameters[tr_type]})
    
    
    #['flip','zoom','shift','rotation']
    def multi_transform(self, img: Union[str, np.ndarray] = None, 
                        chain_transform: Optional[List[str]] = None,
                         params: Optional[dict] = None, update: bool = True) -> np.ndarray:

        """
        Applies a chain of multiple transformations to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to transform. Uses the class's internal image data if None.
        chain_transform : list, optional
            A list of transformation names to apply.
        params : dict, optional
            Parameters for each transformation in the chain.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """
        
        # Selecting transformations if not provided
        if chain_transform is None:
            chain_transform = self._multitr_chain if self._multitr_chain is not None else self._select_random_transforms()
                 
        if img is None:
            img = self.img_data

        imgtr = copy.deepcopy(img)
        augmentedsuffix = {}
        print(chain_transform)
        for transform_name in chain_transform:
            if params is None:
                imgtr = perform_kwargs(self._run_default_transforms[transform_name],
                     img = imgtr,
                     update = False)
            else:
                
                imgtr = perform(self._run_default_transforms[transform_name],
                     imgtr,
                     params[transform_name], False)
            #if update:
            augmentedsuffix[transform_name] = self._transformparameters[transform_name]
        
        self._transformparameters['multitr'] = augmentedsuffix
         
        if update:
            
            self.updated_paramaters(tr_type = 'multitr')
            self._new_images['multitr'] = imgtr

        return imgtr


    def apply_transformation(self, transform_func, img=None, transform_param=None, 
                             transform_name=None, update=True):
        """
        Applies a specific image transformation function.

        Parameters:
        ----------
        transform_func : function
            The specific transformation function to apply.
        img : ndarray, optional
            The image to be transformed. If None, uses the class's internal image data.
        transform_param : various types, optional
            The parameter specific to the transformation.
        transform_name : str, optional
            The name of the transformation (for internal tracking).
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The transformed image.
        """

        if img is None:
            img = self.img_data

        if transform_param is None and transform_name:
            transform_param = self._random_parameters.get(transform_name, None)

        # Validate that the function is callable
        if not callable(transform_func):
            raise ValueError("Provided function is not callable.")
        img_transformed = transform_func(img, transform_param)

        if update and transform_name:
            self._transformparameters[transform_name] = transform_param
            self.updated_paramaters(tr_type=transform_name)
            self._new_images[transform_name] = img_transformed

        return img_transformed

    def diff_gaussian_image(self, img: Union[str, np.ndarray] = None, 
                            high_sigma: Optional[float] = None, update: bool = True):
        """
        Applies a differential Gaussian filter to the image.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be processed. If None, uses the class's internal image data.
        high_sigma : float, optional
            The standard deviation for the Gaussian kernel. If None, a random value is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray
            The image after applying the differential Gaussian filter.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        if high_sigma is None:
            high_sigma = self._random_parameters['gaussian']

        high_sigma = max(min(high_sigma, 100), 0.1)
        
        imgtr,_ = diff_guassian_img(img,high_sigma = high_sigma)
        self._transformparameters['gaussian'] = high_sigma
        
        if update:
            
            self.updated_paramaters(tr_type = 'gaussian')
            self._new_images['gaussian'] = imgtr

        return imgtr
    
    def perspective_image(self, img: np.ndarray = None, perspective_x: float = None, perspective_y: float = None,update:bool = True):
        """
        Apply a perspective transformation to an image.

        Parameters:
        ----------
        img : np.ndarray
        The input image to be transformed.
        perspective_x : float, optional
            The perspective transformation factor along the x-axis.
        perspective_y : float, optional
            The perspective transformation factor along the y-axis.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        if perspective_x is None:
            perspective_x, _ = self._random_parameters['perspective']
        if perspective_y is None:
            _, perspective_y = self._random_parameters['perspective']

        
        imgtr = perspective_transform(img,perspective_x = perspective_x, perspective_y=perspective_y)
        self._transformparameters['perspective'] = [perspective_x, perspective_y]
        
        if update:
            
            self.updated_paramaters(tr_type = 'perspective')
            self._new_images['perspective'] = imgtr

        return imgtr
    
    def shear_image(self, img: np.ndarray = None, shear_x: float = None, shear_y:float = None,update:bool = True):
        """
        Shear the given image by a specified angle.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be rotated. If None, uses the class's internal image data.
        shear_x : float, optional
            The shear factor for shear the image in the x axis. the values must be between 0 to 1
        shear_y : float, optional
            The shear factor for shear the image in the y axis. the values must be between 0 to 1
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        if shear_x is None:
            shear_x, _ = self._random_parameters['shear']
        if shear_y is None:
            _, shear_y = self._random_parameters['shear']

        
        imgtr = shear_image(img,shear_x = shear_x, shear_y=shear_y)
        self._transformparameters['shear'] = [shear_x, shear_y]
        
        if update:
            
            self.updated_paramaters(tr_type = 'shear')
            self._new_images['shear'] = imgtr

        return imgtr
    
    def rotate_image(self, img = None, angle = None, update = True):
        """
        Rotates the given image by a specified angle.

        Parameters:
        ----------
        img : ndarray, optional
            The image to be rotated. If None, uses the class's internal image data.
        angle : int, optional
            The angle in degrees for rotating the image. If None, a random angle is chosen based on class parameters.
        update : bool, optional
            If True, updates the class's internal state with the result.

        Returns:
        -------
        ndarray:
            The rotated image.

        Raises:
        ------
        ValueError:
            If the input image is not in the expected format or dimensions.
        """
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        if angle is None:
            angle = self._random_parameters['rotation']

        
        imgtr = image_rotation(img,angle = angle)
        self._transformparameters['rotation'] = angle
        
        if update:
            
            self.updated_paramaters(tr_type = 'rotation')
            self._new_images['rotation'] = imgtr

        return imgtr

    def flip_image(self, img = None, flipcode = None, update = True):

        if img is None:
            img = copy.deepcopy(self.img_data)
        if flipcode is None:
            flipcode = self._random_parameters['flip']

        
        imgtr = image_flip(img,flipcode = flipcode)
        
        self._transformparameters['flip'] = flipcode
        
        if update:
            
            self.updated_paramaters(tr_type = 'flip')
            self._new_images['flip'] = imgtr

        return imgtr

    def expand_image(self, img = None, ratio = None, update = True):
        if ratio is None:
            ratio = self._random_parameters['zoom']
            
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        imgtr = image_zoom(img, zoom_factor=ratio)
        
        self._transformparameters['zoom'] = ratio
        if update:
            
            self.updated_paramaters(tr_type = 'zoom')
            self._new_images['zoom'] = imgtr

        return imgtr
    

    def shift_ndimage(self,img = None, xshift  = None, yshift = None, update = True,
                      max_displacement = None):

        
        if max_displacement is None and xshift is None:
            max_displacement = (self._random_parameters['shift'])/100
        if img is None:
            img = copy.deepcopy(self.img_data)


        imgtr, displacement =  randomly_displace(img, 
                                                 maxshift = max_displacement, 
                                                 xshift = xshift, yshift = yshift)
        
        self._transformparameters['shift'] = displacement
        if update:
            
            self.updated_paramaters(tr_type = 'shift')
            self._new_images['shift'] = imgtr#.astype(np.uint8)

        return imgtr#.astype(np.uint8)
    
    def change_illumination(self, img = None, illuminationparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if illuminationparams is None:
            illuminationparams = self._random_parameters['illumination']
        
        
        cv2img = cv2_array_type(img)
           
        imgtr = illumination_shift(cv2img,valuel = illuminationparams)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
            
        self._transformparameters['illumination'] = illuminationparams
        if update:
            
            self.updated_paramaters(tr_type = 'illumination')
            self._new_images['illumination'] = imgtr

        return imgtr
    
    def clahe(self, img= None, thr_constrast = None, update = True):

        if thr_constrast is None:
            thr_constrast = self._random_parameters['clahe']/10
        
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        cv2img = cv2_array_type(img)
        
        imgtr,_ = clahe_img(cv2img, clip_limit=thr_constrast)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
            
        self._transformparameters['clahe'] = thr_constrast
        if update:
            
            self.updated_paramaters(tr_type = 'clahe')
            self._new_images['clahe'] = imgtr
            
        return imgtr

    def hsv(self, img = None, hsvparams =None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        if hsvparams is None:
            hsvparams = self._random_parameters['hsv']
        
        cv2img = cv2_array_type(img)
        
        imgtr,_ = shift_hsv(cv2img,hue_shift=hsvparams[0], sat_shift = hsvparams[1], val_shift = 0)
        
        if img.shape[0] == 3:
            imgtr = imgtr.swapaxes(2,1).swapaxes(1,0)
        if np.nanmax(img) < 2:
            imgtr = imgtr/255.
        
        self._transformparameters['hsv'] = hsvparams
        if update:
            
            self.updated_paramaters(tr_type = 'hsv')
            self._new_images['hsv'] = imgtr

        return imgtr
    
    def random_augmented_image(self,img= None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        imgtr = copy.deepcopy(img)
        augfun = random.choice(list(self._run_default_transforms.keys()))
        
        imgtr = perform_kwargs(self._run_default_transforms[augfun],
                     img = imgtr,
                     update = update)

        return imgtr

    def _transform_as_ids(self, tr_type):

        if type (self.tr_paramaters[tr_type]) ==  dict:
            paramsnames= ''
            for j in list(self.tr_paramaters[tr_type].keys()):
                paramsnames += 'ty_{}_{}'.format(
                    j,
                    summarise_trasstring(self.tr_paramaters[tr_type][j]) 
                )

        else:
            paramsnames = summarise_trasstring(self.tr_paramaters[tr_type])

        return '{}_{}'.format(
                tr_type,
                paramsnames
            )
    
    def augmented_names(self):
        transformtype = list(self.tr_paramaters.keys())
        augmentedsuffix = {}
        for i in transformtype:
            
            augmentedsuffix[i] = self._transform_as_ids(i)

        return augmentedsuffix
    
    


class MultiChannelImage(ImageAugmentation):
    """
    A class to handle and transform multi-channel images, such as RGB or multispectral images.
    This class extends basic functionality with multiple image-specific transformations and 
    handles both single and batch operations on image data.

    Attributes
    ----------
    orig_imgname : str
        The original name or identifier of the image.
    mlchannel_data : np.ndarray
        The multi-channel image data stored as a 3D numpy array.
    tr_paramaters : dict
        A dictionary mapping transformation types to their respective parameters.
    _new_images : dict
        A dictionary containing transformed image data for each type of transformation.

    Methods
    -------
    Various transformation methods like `rotate_multiimages`, `flip_multiimages`, etc.,
    are provided to apply respective transformations to multi-channel images.
    """
    
    
    def __init__(self, img: np.ndarray = None, img_id: str = None, 
                 channels_order: str = 'first', transforms: List[str] = None, **kwargs) -> None:
        """
        Initializes the MultiChannelImage class with a multi-channel image and additional parameters.

        Parameters
        ----------
        img : np.ndarray
            The multi-channel image data to be processed.
        img_id : str, optional
            An identifier or name for the image. Default is None.
        channels_order : str, optional
            Specifies the ordering of the channels in the input image. Accepted values are 'first' or 'last'.
            'first' indicates that channels are in the first dimension, 'last' indicates that channels are in the last dimension.
        transforms : list of str, optional
            A list of transformation names to be available for the image. If None, all default transformations are used.
        **kwargs : dict
            Additional keyword arguments.
        """
        
        if channels_order not in ['first', 'last']:
            raise ValueError("channels_order must be either 'first' or 'last'.")

        ### set random transforms
        self._availableoptions = list(self._run_multichannel_transforms.keys())+['raw'] if transforms is None else transforms+['raw']

        self.scaler = None
        self.orig_imgname = img_id or "image"
        if img is not None:
            self._initimg = img[0] if channels_order == 'first' else img[:,:,0]
        else:
            self._initimg = None
        self.mlchannel_data = img

        super().__init__(self._initimg, **kwargs)
    
    def __call__(self, img: np.ndarray, transform, **kwargs):
        
        imgval = self._transform_multichannel(img.copy(),transform, **kwargs)
        return imgval
                
    
    @property
    def _run_multichannel_transforms(self):
        return  {
                'rotation': self.rotate_multiimages,
                'zoom': self.expand_multiimages,
                'illumination': self.diff_illumination_multiimages,
                'shear': self.shear_multiimages,
                'shift': self.shift_multiimages,
                #'multitr': self.multitr_multiimages,
                'flip': self.flip_multiimages,
                'clahe': self.clahe_multiimages,
                'hsv': self.hsv_multiimages,
                'perspective': self.perspective_multiimages
            }
    

    @property
    def imgs_data(self):
        imgdata = {'raw': self.mlchannel_data}
        augementedimgs = self._augmented_images
        if len(list(augementedimgs.keys()))> 0:
            for datatype in list(augementedimgs.keys()):
                currentdata = {datatype: augementedimgs[datatype]}
                imgdata.update(currentdata)

        return imgdata
    
    @property
    def images_names(self):
        imgnames = {'raw': self.orig_imgname}
        augementednames = self.augmented_names()
        if len(list(augementednames.keys()))> 0:
            for datatype in list(augementednames.keys()):
                currentdata = {datatype: '{}_{}'.format(
                    imgnames['raw'],
                    augementednames[datatype])}
                imgnames.update(currentdata)

        return imgnames
    
    def _scale_multichannels_data(self,img, method = 'standarization'):
        """
        Scales multi-channel image data using the specified method.

        Parameters:
        ----------
        img : numpy.ndarray
            The multi-channel image data to be scaled.
        method : str, optional
            The scaling method to apply. Supported methods: 'standarization'.

        Returns:
        -------
        numpy.ndarray
            The scaled multi-channel image data.
        """
        
        if self.scaler is not None:
                        
            if method == 'standarization':
                
                datascaled = []
                for z in range(self.mlchannel_data.shape[0]):
                    datascaled.append(standard_scale(
                            img[z].copy(),
                            meanval = self.scaler[z][0], 
                            stdval = self.scaler[z][1]))

        else:
            datascaled = img.copy()
            
        return datascaled
    
    def _tranform_channelimg_function(self, img: np.ndarray, tr_name: str) -> np.ndarray:
        """
        Transforms a single channel image based on the specified transformation name.

        Parameters
        ----------
        img : np.ndarray
            The single channel image data to be transformed.
        tr_name : str
            The name of the transformation to apply.

        Returns
        -------
        np.ndarray
            The transformed single channel image.
        """
        
    
        if tr_name == 'multitr':
            params = self.tr_paramaters[tr_name]
            image = self.multi_transform(img=img,
                                chain_transform = list(params.keys()),
                                params= params, update= False)

        else:
            if isinstance(self.tr_paramaters[tr_name],list):
                image =  perform(self._run_default_transforms[tr_name],
                        img,
                        *self.tr_paramaters[tr_name], False)
            else:
                image =  perform(self._run_default_transforms[tr_name],
                        img,
                        self.tr_paramaters[tr_name], False)

        return image

    def _transform_multichannel(self, img=None, tranformid = None,  **kwargs):
        """
        Applies a transformation to each channel of the multi-channel image.

        Parameters:
        ----------
        img : numpy.ndarray, optional
            The multi-channel image data to transform. Uses class's image data if None.
        transform_id : str
            The identifier of the transformation to apply.
        **kwargs
            Additional keyword arguments for the transformation.

        Returns:
        -------
        numpy.ndarray
            The transformed multi-channel image data.
        """
        
        trimgs = img if img is not None else self.mlchannel_data
        #if tranformid == 'multitr':
        
        if tranformid == 'raw':
            return trimgs
        
        if tranformid not in ['illumination', 'clahe', 'hsv']:
            
            imgs= [perform_kwargs(self._run_default_transforms[tranformid],
                        img = trimgs[0],
                        **kwargs)]
            
            imgs = imgs + [self._tranform_channelimg_function(trimgs[i],tranformid) for i in range(1,trimgs.shape[0])]
            
            imgs = np.stack(imgs, axis = 0)
            
        else:
            imgs= perform_kwargs(self._run_default_transforms[tranformid],
                        img = trimgs,
                        **kwargs)
            
        self._new_images[tranformid] = imgs
        
        return imgs

    def shift_multiimages(self, img=None, xshift=None, yshift=None, max_displacement=None,update=True):

        self._new_images['shift'] =  self._transform_multichannel(img=img, 
                    tranformid = 'shift', xshift = xshift, yshift = yshift, max_displacement=max_displacement,update=update)
        return self._new_images['shift']

    def rotate_multiimages(self, img=None, angle=None, update=True):
        self._new_images['rotation'] = self._transform_multichannel(img=img, 
                    tranformid = 'rotation', angle = angle, update=update)
        
        return self._new_images['rotation']
    
    def perspective_multiimages(self, img=None, perspective_x=None, perspective_y=None, update=True):
        self._new_images['perspective'] = self._transform_multichannel(img=img, 
                    tranformid = 'perspective', perspective_x = perspective_x, perspective_y=perspective_y, update=update)
        
        return self._new_images['perspective']
    
    def shear_multiimages(self, img=None, shear_x=None, shear_y=None, update=True):
        self._new_images['shear'] = self._transform_multichannel(img=img, 
                    tranformid = 'shear', shear_x = shear_x, shear_y=shear_y, update=update)
        
        return self._new_images['shear']
    
    def flip_multiimages(self, img=None, flipcode=None, update=True):
        self._new_images['flip'] = self._transform_multichannel(img=img, 
                    tranformid = 'flip', flipcode = flipcode, update=update)
        
        return self._new_images['flip']
    
    def diff_guassian_multiimages(self, img=None, high_sigma=None, update=True):
        self._new_images['gaussian'] = self._transform_multichannel(img=img, 
                    tranformid = 'gaussian', high_sigma = high_sigma, update=update)
        
        return self._new_images['gaussian']
    
    def diff_illumination_multiimages(self, img=None, illuminationparams=None, update=True):
        self._new_images['illumination'] = self._transform_multichannel(img=img, 
                    tranformid = 'illumination', illuminationparams = illuminationparams, update=update)
        
        return self._new_images['illumination']
    
    def hsv_multiimages(self, img= None, hsvparams = None, update = True):
        self._new_images['hsv'] = self._transform_multichannel(img=img, 
                    tranformid = 'hsv', hsvparams = hsvparams, update=update)
        
        return self._new_images['hsv']
    
    def clahe_multiimages(self, img= None, thr_constrast = None, update = True):
        self._new_images['clahe'] = self._transform_multichannel(img=img, 
                    tranformid = 'clahe', thr_constrast = thr_constrast, update=update)
        
        return self._new_images['clahe']
        
    def expand_multiimages(self, img=None, ratio=None, update=True):

        self._new_images['zoom'] = self._transform_multichannel(img=img, 
                    tranformid = 'zoom', ratio = ratio, update=update)
        
        return self._new_images['zoom']


    def multitr_multiimages(self, img=None,  
                        params=None, 
                        update=True):
        
        self._new_images['multitr'] = self._transform_multichannel(
                    img=img, 
                    tranformid = 'multitr', 
                    params=params, update=update)
        
        return self._new_images['multitr']
    
    def apply_transform_chain(self, transformations: List[str] = None, n_chains: int = 2) -> np.ndarray:
        """
        Applies a chain of random transformations to the multi-channel image.

        Parameters
        ----------
        transformation: list, optional
            The name of the transformations.
            
        n_chains : int, optional
            The number of transformations to chain. Default is 2.

        Returns
        -------
        np.ndarray
            The transformed multi-channel image data.
        """
        
        self._transformparameters = {}
        transformations = self._select_random_transforms(n_chains=n_chains) if transformations is None else transformations

        imgval = self.mlchannel_data.copy()
        for trfun in transformations:
            imgval = self._run_multichannel_transforms[trfun](imgval)
        
        return imgval 

    def random_transform(self, augfun: str = None, verbose: bool = False, parameters: dict = None) -> np.ndarray:
        """
        Applies a random transformation to the multi-channel image.

        Parameters
        ----------
        augfun : str, optional
            The name of the augmentation function to use. If None, a random function is chosen.
        verbose : bool, optional
            If True, prints the applied transformation name.
        parameters : dict, optional
            Additional parameters for the transformation.

        Returns
        -------
        np.ndarray
            The randomly transformed multi-channel image data.
        """        
        if augfun is None:
            augfun = random.choice(self._availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in self._availableoptions:
            print(f"""That augmentation option is not in default parameters {self._availableoptions},
                     no transform was applied""")
            augfun = 'raw'
        
        imgtr = self.mlchannel_data.copy() if augfun == 'raw' else perform_kwargs(self._run_multichannel_transforms[augfun])
        imgtr = self._scale_multichannels_data(imgtr)

        if verbose:
            print(f'{augfun} was applied')
            
        return imgtr
    
    
class TranformPairedImages(MultiChannelImage):

        
    def __init__(self, multitr_transform = ['zoom','flip','rotation']) -> None:
        
        #self.img1 = img1.copy()
        #self.img2 = img2.copy()
        self._list_not_transform = ['illumination','clahe','hsv']
        
        self._multitr_options = multitr_transform
    
    def __call__(self, img1, img2, **kwargs):
        """_summary_

        Args:
            img1 (_type_): numpy image dimension HWC
            img2 (_type_): HWC

        Returns:
            _type_: _description_
        """
        self.img1 = img1.copy()
        self.img2 = img2.copy()
        super().__init__(self.img1.copy(), **kwargs)
        self._multitr_chain = self._multitr_options
        img1tr, img2tr = self.random_paired_transform()
        
        return img1tr, img2tr
        
        
    
    
    def random_paired_transform(self, augfun = None, verbose = False):
        
        img1tr = self.random_transform(augfun=augfun,verbose = verbose)
        listtrfunction = list(self.tr_paramaters.keys())
        if len(listtrfunction) > 0:
            trname = listtrfunction[-1]
            img2tr = self._tranform_img2_function(trname) if trname not in self._list_not_transform else self.img2 
        else:
            img2tr = self.img2 
        return [img1tr, img2tr]
        


    def _tranform_img2_function(self, tr_name):
        """compute transformation for targets"""
        if len(self.img2.shape) == 1:
            trmask =  np.expand_dims(np.zeros(self.img1.shape[:2]), axis = 0)    
        
        else:
            trmask = np.zeros(self.img2.shape)
        
        for i,mask in enumerate(self.img2):
            
            if tr_name == 'multitr':
                
                params = self.tr_paramaters[tr_name]
                paramslist = [partr for partr in list(params.keys()) if partr not in self._list_not_transform]
                filteredparams = {}
                for partr in paramslist:
                    filteredparams[partr] = params[partr]
                    
                imgtr = mask.copy()

                for partr in paramslist:
                    #print(partr)
                    imgtr = perform(self._run_default_transforms[partr],
                            imgtr,
                            params[partr], False)
                
                trmask[i] = imgtr

            else:
                trmask[i]  =  perform(self._run_default_transforms[tr_name],
                        mask.copy(),
                        self.tr_paramaters[tr_name], False)

        return trmask


def scale_mtdata(npdata,
                 features,
                       scaler=None, 
                       scale_z = True,
                       name_3dfeature = 'z', 
                       method = 'minmax',
                       shapeorder = 'DCHW'):

    datamod = npdata.copy()
    if shapeorder == 'DCHW':
        datamod = datamod.swapaxes(0,1)


    if scaler is not None:
        
        datascaled = []

        for i, varname in enumerate(features):
            if method == 'minmax':
                scale1, scale2 = scaler
                if scale_z and varname == name_3dfeature:
                        datascaled.append(minmax_scale(
                            datamod[i],
                            minval = scale1[varname], 
                            maxval = scale2[varname]))

                elif varname != name_3dfeature:
                    scale1[varname]
                    datascaled.append(minmax_scale(
                            datamod[i],
                            minval = scale1[varname], 
                            maxval = scale2[varname]))
                else:
                    datascaled.append(datamod[i])
            
            elif method == 'normstd':
                scale1, scale2 = scaler
                if scale_z and varname == name_3dfeature:
                        datascaled.append(standard_scale(
                            datamod[i],
                            meanval = scale1[varname], 
                            stdval = scale2[varname]))

                elif varname != name_3dfeature:
                    scale1[varname]
                    datascaled.append(standard_scale(
                            datamod[i],
                            meanval = scale1[varname], 
                            stdval = scale2[varname]))
                else:
                    datascaled.append(datamod[i])

    if shapeorder == 'DCHW':
        datascaled = np.array(datascaled).swapaxes(0,1)
    else:
        datascaled = np.array(datascaled)

    return datascaled


class MultiTimeTransform(MultiChannelImage):
    """
    Handles transformations on multitemporal and multichannel image data, allowing for various image augmentation
    techniques to be applied sequentially or randomly.

    Parameters
    ----------
    data : np.ndarray, optional
        Multitemporal data array to be transformed.
    onlythesedates : list, optional
        Specific indices of dates to be used from the data.
    img_id : Any, optional
        Identifier for the image data.
    formatorder : str, optional
        Order of dimensions in the data array, default is 'DCHW' (Depth, Channel, Height, Width).
    channelslast : bool, optional
        Whether the channels are in the last dimension, default is False.
    removenan : bool, optional
        Whether to replace NaN values with zeros, default is True.
    image_scaler : Any, optional
        Scaler object or parameters for image scaling.
    scale_3dimage : bool, optional
        Whether scaling should be applied to 3D images, default is False.
    name_3dfeature : str, optional
        Name of the 3D feature if 3D scaling is used, default is 'z'.
    scale_method : str, optional
        Method of scaling to apply, default is 'minmax'.
    transform_options : Any, optional
        Additional transformation options or configurations.

    Attributes
    ----------
    scaler_params : dict or None
        Parameters for scaling the image data.
    npdata : np.ndarray
        The original multitemporal data array.
    _raw_img : np.ndarray
        A deep copy of the original data for restoration purposes.
    _initdate : np.ndarray
        The initial date image from the data array for reference transformations.
    _formatorder : str
        Internal format order used for processing, fixed as 'CDHW'.
    _orig_formatorder : str
        Original format order of the input data.
    """
    
    
    def __init__(self, data: Optional[np.ndarray] = None, onlythesedates: Optional[List[int]] = None,
                 img_id: Optional[Any] = None, formatorder: str = 'DCHW',
                 channelslast: bool = False, removenan: bool = True,
                 image_scaler: Optional[Any] = None, scale_3dimage: bool = False,
                 name_3dfeature: str = 'z', scale_method: str = 'minmax',
                 transform_options: Optional[Any] = None, **kwargs) -> None:
        
        """
        transform multitemporal data

        Args:
            data (_type_, optional): _description_. Defaults to None.
            onlythesedates (_type_, optional): _description_. Defaults to None.
            img_id (_type_, optional): _description_. Defaults to None.
            formatorder (str, optional): _description_. Defaults to 'DCHW'.
            channelslast (bool, optional): _description_. Defaults to False.
            removenan (bool, optional): _description_. Defaults to True.
            image_scaler (_type_, optional): _description_. Defaults to None.
            scale_3dimage (bool, optional): _description_. Defaults to False.
            name_3dfeature (str, optional): _description_. Defaults to 'z'.
            scale_method (str, optional): _description_. Defaults to 'minmax'.
        """

        if image_scaler is not None:
            self.scaler_params = {
                'scaler': image_scaler,
                'method': scale_method,
                'scale_3dimage': scale_3dimage,
                'name_3dfeature': name_3dfeature
            }
        else:
            self.scaler_params = None

        self.npdata = copy.deepcopy(data)

        
        if removenan:
            self.npdata[np.isnan(self.npdata)] = 0
        self._orig_formatorder = formatorder
        self._formatorder = "CDHW"
        
        if self._orig_formatorder == "DCHW":
            self.npdata = self.npdata.swapaxes(1,0)
            channelsorder = 'first'
        elif self._orig_formatorder == "CDHW":
            channelsorder = 'first'
        elif channelslast:
            channelsorder = 'last'
        else:
            channelsorder = 'first'
            
        if onlythesedates is not None:
            self.npdata = self.npdata[:,onlythesedates,:,:]

        self._raw_img = copy.deepcopy(self.npdata)

        #if image_scaler is not None:
        #    self.npdata = scale_mtdata(self,[image_scaler[0],image_scaler[1]],
        #                            scale_z = scale_3dimage, name_3dfeature = name_3dfeature, method=scale_method)

        self._initdate = copy.deepcopy(self.npdata[:,0,:,:])

        MultiChannelImage.__init__(self,img = self._initdate, img_id= img_id, 
                                   channels_order = channelsorder, transforms= transform_options,**kwargs)


    @property
    def _run_random_choice(self):
        """
        A property to define a dictionary mapping transformation types to their corresponding methods.

        Returns
        -------
        Dict[str, callable]
            A dictionary linking transformation names to methods.
        """
        return  {
                'rotation': self.rotate_tempimages,
                'zoom': self.expand_tempimages,
                'shift': self.shift_multi_tempimages,
                'illumination': self.illumination_tempimages,
                'gaussian': self.diff_guassian_tempimages,
                'multitr': self.multtr_tempimages,
                'flip': self.flip_tempimages,
                'shear': self.shear_tempimages
            }

    def _scale_image(self, img):
        

        if self.scaler_params is not None:
            ## data D C X Y
            img= scale_mtdata(img,self.features,
                                self.scaler_params['scaler'],
                                scale_z = self.scaler_params['scale_3dimage'], 
                                name_3dfeature = self.scaler_params['name_3dfeature'], 
                                method=self.scaler_params['method'],
                                shapeorder = self._formatorder)

        return img


    def _multi_timetransform(self, tranformn, changeparamsmlt = False, **kwargs):
        imgs= [perform_kwargs(self._run_multichannel_transforms[tranformn],
                     img = self._initdate,
                     **kwargs)]

        for i in range(1,self.npdata.shape[1]):

            if tranformn != 'multitr':
                if changeparamsmlt:
                    params = self._random_parameters[tranformn]
                else:
                    params = self.tr_paramaters[tranformn]
                
                r = perform(self._run_multichannel_transforms[tranformn],
                               self.npdata[:,i,:,:],
                               params,
                               False,
                               )
            else:
                
                r = perform_kwargs(self._run_multichannel_transforms[tranformn],
                               img=self.npdata[:,i,:,:],
                               params = self.tr_paramaters[tranformn],
                               update = False,
                               )
            imgs.append(r)

        imgs = np.stack(imgs,axis=1)

        #imgs = self._scale_image(imgs)
        imgs = self._return_orig_format(imgs)
        self.npdata = copy.deepcopy(self._raw_img)
        
        return imgs

    def shift_multi_tempimages(self, shift=None, max_displacement=None):
        return self._multi_timetransform(tranformn = 'shift',
                                        shift= shift, 
                                        max_displacement= max_displacement)

    def diff_guassian_tempimages(self, high_sigma=None):
        return self._multi_timetransform(tranformn = 'gaussian', high_sigma = high_sigma)
    
    def expand_tempimages(self, ratio=None):
        return self._multi_timetransform(tranformn = 'zoom', ratio = ratio)

    def shear_tempimages(self, shear_x=None, shear_y = None):
        return self._multi_timetransform(tranformn = 'shear', shear_x = shear_x, shear_y = shear_y)
    
    def rotate_tempimages(self, angle=None):
        return self._multi_timetransform(tranformn = 'rotation', angle = angle)
    
    def flip_tempimages(self, flipcode=None):
        return self._multi_timetransform(tranformn = 'flip', flipcode = flipcode)
        
    def illumination_tempimages(self, illuminationparams=None):
        return self._multi_timetransform(tranformn = 'illumination', changeparamsmlt=True,illuminationparams = illuminationparams)

    def multtr_tempimages(self, img=None, chain_transform=['flip','zoom','shift', 'rotation'], params=None):
        return self._multi_timetransform(tranformn = 'multitr', 
                                         chain_transform= chain_transform, 
                                         params = params)

        
    def random_multime_transform(self, augfun = None, verbose = False):
        availableoptions = self._availableoptions
        
        if augfun is None:
            augfun = random.choice(availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in availableoptions:
            print(f"""that augmentation option is not into default parameters {availableoptions},
                     no transform was applied""")
            augfun = 'raw'

        if augfun == 'illumination' and self.npdata.shape[0]!=3:
            augfun = 'raw'
            
        if augfun == 'raw':
            imgtr = self.npdata#.swapaxes(0,1)
            imgtr = self._return_orig_format(imgtr)
        else:
            imgtr = perform_kwargs(self._run_random_choice[augfun])

        if verbose:
            print('{} was applied'.format(augfun))
        
        
        #imgtr = self._return_orig_format(imgtr)
        
        return imgtr

    def _return_orig_format(self, imgtr):

        if self._orig_formatorder == "DCHW":
            imgtr = np.einsum('CDHW->DCHW', imgtr)
        
        if self._orig_formatorder == "HWCD":
            imgtr = np.einsum('CDHW->HWCD', imgtr)
        
        return imgtr


#### Image with bounding boxes




def export_masked_image(image, ground_box, filename):
    bb_int = percentage_to_bb(ground_box['detection_boxes'][0],
                              (image.shape[1], image.shape[0]))

    image = Image.fromarray(image)
    image = image.crop(box=bb_int[0])

    image.save(filename)
    print("Image saved: {}".format(filename))

