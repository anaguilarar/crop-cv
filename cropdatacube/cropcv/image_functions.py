from PIL import Image, ImageOps, ImageFilter

import numpy as np
from six import BytesIO
import io
from skimage.transform import SimilarityTransform
from skimage.filters import difference_of_gaussians
from skimage.draw import line_aa
from skimage.transform import warp
from pathlib import Path
from itertools import product
from math import cos, sin, radians
import os
import numpy as np
import random
import math
from scipy import ndimage
from scipy.spatial import ConvexHull
import scipy.signal

import cv2
import warnings
from typing import Tuple, Optional, Union
from sklearn.impute import KNNImputer


def fill_na_values(mltimage: np.ndarray, n_neighbors: int = 7) -> np.ndarray:
    """
    Fill NaN values in a multi-dimensional image using k-Nearest Neighbors (k-NN) imputation.

    Parameters
    ----------
    mltimage : np.ndarray
        Multi-dimensional array with shape (C, T, H, W), where C is the number of channels, 
        T is the number of time steps, H is the height, and W is the width.
    n_neighbors : int, optional
        Number of neighboring samples to use for imputation. Default is 7.

    Returns
    -------
    np.ndarray
        New array with NaN values filled using k-NN imputation.
    """
    impmodel = KNNImputer(n_neighbors=n_neighbors)
    newimg = mltimage.copy()
    for c in range(mltimage.shape[0]):
        for t in range(mltimage.shape[1]):
            if np.isnan(mltimage[c, t]).any():
                newimg[c,t] = impmodel.fit_transform(mltimage[c,t])

    return newimg   


def apply_color_alpha(image, mask, color, alpha=0.5):
    """
    Apply color with alpha blending to the image based on the mask.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    mask : np.ndarray
        Mask indicating where the color should be applied.
    color : tuple
        RGB color tuple (0-1 range) to be applied.
    alpha : float, optional
        Alpha value for blending (default is 0.5).

    Returns:
    --------
    np.ndarray
        Image with color applied based on the mask.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def resize_2dimg(image: np.ndarray, newx: int, newy: int, flip: bool = True, 
                 interpolation: str = 'bilinear', blur: bool = False, 
                 kernelsize: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Resizes a 2D image using specified dimensions and options for flipping and blurring.

    Parameters
    ----------
    image : np.ndarray
        The input image array to be resized.
    newx : int
        The new width of the image.
    newy : int
        The new height of the image.
    flip : bool, optional
        A boolean flag to indicate whether the image should be flipped vertically after resizing. Default is True.
    interpolation : str, optional
        The method of interpolation to be used for resizing. Options are 'bilinear', 'bicubic', 'nearest'.
        Default is 'bilinear'.
    blur : bool, optional
        A boolean flag to indicate whether a Gaussian blur should be applied to the resized image. Default is False.
    kernelsize : Tuple[int, int], optional
        The size of the kernel to be used for the Gaussian blur. Default is (5, 5).

    Returns
    -------
    np.ndarray
        The processed image array after resizing, optional flipping, and optional blurring.

    Notes
    -----
    Uses OpenCV for resizing and Gaussian blur, and PIL for flipping the image if required.
    """
    intmethod = cv2.INTER_LINEAR
    if interpolation == 'bilinear':
        intmethod =  cv2.INTER_LINEAR
    if interpolation == 'bicubic':
        intmethod =  cv2.INTER_CUBIC
    if interpolation == 'nearest':
        intmethod = cv2.INTER_NEAREST
    imageres = cv2.resize(image, (newx, newy), interpolation= intmethod)

    if flip:
        imageres = Image.fromarray(imageres)
        imageres = np.array(ImageOps.flip(imageres))
    if blur:
        imageres = cv2.GaussianBlur(imageres,kernelsize,0)

    return imageres

def resize_npimage(image, size: Tuple[int]):
    """Resize image

    Args:
        image (np.array): numpy array with dimensions H W C 
        size (tuple): new height and width values.

    Returns:
        image (np.array): tranformed image with new size H W C
    """
    im_width = size[1]
    im_height = size[0]
    
    pil_image = Image.fromarray(image)

    img = pil_image.resize((im_width, im_height), Image.Resampling.LANCZOS)
    
    if len(image.shape) == 3:
        nchannels = image.shape[2]
        img = np.array(img).reshape(
        (im_height, im_width, nchannels))
    if len(image.shape) == 2:
        img = np.array(img).reshape(
        (im_height, im_width))
          
    return img


def read_image_as_numpy_array(path, scale_factor=None,
                                size=None):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(io.open(path, "rb", buffering=0).read()))
    image = image.convert("RGB") if path.endswith('.png') else image
    
    (im_width, im_height) = image.size
    if scale_factor is not None:
        im_width = int(im_width * scale_factor / 100)
        im_height = int(im_height * scale_factor / 100)
        image = image.resize((im_width, im_height))
    if size is not None:
        im_width = size[1]
        im_height = size[0]
        image = image.resize((im_width, im_height))
    if path.endswith('.tiff'):
        nlen = len(image.getbands())-1 if 'A' in image.getbands() else len(image.getbands())
        image = np.array(image, dtype='uint8')[:,:,:nlen]
    else:
        image = np.array(image.getdata(), dtype='uint8').reshape((im_height, im_width, 3))

    return image
    

def cv2_array_type(arrayimage):
    
    imgcvtype = arrayimage.copy()
    if not np.max(imgcvtype)>120:
        imgcvtype = imgcvtype*255
        
    imgcvtype[imgcvtype<0] = 0
    imgcvtype[imgcvtype>255] = 255
    imgcvtype = imgcvtype.astype(np.uint8)

    if imgcvtype.shape[0]<=3:
        imgcvtype = imgcvtype.swapaxes(0,1).swapaxes(1,2)
        
    return imgcvtype

def pad_images(image: np.ndarray, padding_factor: int = 2):
    """
    Pad images with zeros around the edges.

    Parameters:
    -----------
    image : np.ndarray
        Array data representing the image. Dimension order HWC.
    padding_factor : int, optional
        Number of pixels to pad around the edges. Defaults to 2.

    Returns:
    --------
    np.ndarray
        Padded image array.
    """
    # Calculate new shape after padding
    padded_shape = (image.shape[0] + padding_factor, image.shape[1] + padding_factor) + image.shape[2:]

    # Create padded image array
    padded_mask = np.zeros(padded_shape, dtype=image.dtype)
    # Copy original image into padded array
    padded_mask[padding_factor//2:-padding_factor//2, 
                padding_factor//2:-padding_factor//2] = image
    
    #contours = find_contours(padded_mask, 0.5)
    return padded_mask



def img_reduction(img: np.ndarray, perc_reduction: float = 0.01) -> np.ndarray:
    """
    Reduces the input image by a specified percentage, cropping towards the center.

    Parameters
    ----------
    img : np.ndarray
        The input image as a 2D numpy array.
    perc_reduction : float, optional
        The percentage of the image to reduce, expressed as a decimal (0 to 1),
        where 0.01 corresponds to a 1% reduction and 0.1 corresponds to a 10% reduction.
        Default is 0.01, meaning a 1% reduction from each side.

    Returns
    -------
    np.ndarray
        The cropped image as a numpy array.

    Notes
    -----
    - The reduction percentage reduces the image equally from all sides towards the center.
    - If the reduction percentage is set to a value that exceeds half the size of the image,
      the reduction will be capped at the image's center point.
    """
    if not 0 <= perc_reduction < 0.5:
        raise ValueError("perc_reduction must be between 0 and 0.5")
    
    perc_reduction = 1-perc_reduction
    xp,yp = img.shape[0]//2,img.shape[1]//2
    y = img.shape[1]
    x = img.shape[0]
    redy = int(y*perc_reduction/2)
    redx = int(x*perc_reduction/2)
    lc = int((xp-redx) if (xp-redx) > 0 else 0)
    rc = int((xp+redx) if (x-(xp+redx)) > 0 else x)
    bc = int((yp-redy) if (yp-redy) > 0 else 0)
    tc = int((yp+redy) if (y-(yp+redy)) > 0 else y)
    
    return img[bc:tc,lc:rc]


def clip_images_using_boundingbox(image, bounding_box, padding_factor = 2,
                                  bbtype = 'xyxy') -> np.ndarray:   
    """
    Clip images using a bounding box.

    Parameters:
    -----------
    image : np.ndarray
        Array data representing the image. Dimension order HWC.
    bounding_box : Tuple[int, int, int, int]
        Bounding box coordinates (x1, y1, x2, y2).
    padding_factor : int, optional
        Number of pixels to pad around the edges. Defaults to 2.
    bbtype : str, optional
        Bounding box type. Defaults to 'xyxy'.    

    Returns:
    --------
    np.ndarray
        Clipped image array.
    """
    
    if bbtype == 'xyxy':
            x1,y1,x2,y2 = bounding_box
    elif bbtype == 'xxyy':
        x1,x2,y1,y2 = bounding_box
            
    height, width = image.shape[:2]
    
    # Calculate new shape after padding
    if padding_factor>0:
        pad_height1, pad_width1 = int(padding_factor) // 2, int(padding_factor) //2
    else:
        pad_height1, pad_width1 = 0,0
        
    newy1 = 0 if (y1 - pad_height1)<0 else (y1 - pad_height1)
    newy2 = height if (y2 + pad_height1)>height else (y2 + pad_height1)
    newx1 = 0 if (x1 - pad_width1)<0 else (x1 - pad_width1)
    newx2 = width if (x2 + pad_width1)>width else (x2 + pad_width1)
    
    clippedimage = image[newy1:newy2, newx1:newx2] 
    
    return clippedimage

def clip_image_usingbb(image: np.ndarray, bounding_box: Tuple[int, int, int, int],
                    bbtype: str = 'xyxy', padding: Optional[int] = None,
                    paddingwithzeros: bool = True) -> np.ndarray:
    """
    Clips an image using a bounding box.

    Parameters:
    -----------
    image : np.ndarray
        Input image array. Dimension order HWC.
    bounding_box : Tuple[int, int, int, int]
        Bounding box coordinates (x1, y1, x2, y2).
    bbtype : str, optional
        Bounding box type. Defaults to 'xyxy'.
    padding : int, optional
        Number of pixels to pad around the edges. Defaults to None.
    padding_with_zeros : bool, optional
        Whether to pad with zeros. If False, the image will be clipped. Defaults to True.

    Returns:
    --------
    np.ndarray
        Clipped or padded image array.
    """

    if bbtype == 'xyxy' or bbtype ==  'xminyminxmaxymax':
        x1,y1,x2,y2 = bounding_box
    elif bbtype == 'xxyy':
        x1,x2,y1,y2 = bounding_box
        
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
    if padding:
        if paddingwithzeros:
            clipped_image = image[y1:y2,x1:x2] 
            #
            clipped_image = pad_images(clipped_image, padding_factor = padding)
        else:
            clipped_image = clip_images_using_boundingbox(image, bounding_box, padding_factor=padding, bbtype=bbtype)
    
    else:
        clipped_image = image[y1:y2,x1:x2] 
        
    return clipped_image


def contours_from_image(image, kernel_size: Tuple[int, int] = (7, 7),
                        lowerlimit: int = 60, upperlimit: int = 255,
                        areathreshold: float = 0.85):
    """
    Extract contours from an image.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    kernel_size : Tuple[int, int], optional
        Size of the structuring element used for morphological operations. Defaults to (7, 7).
    lowerlimit : int, optional
        Lower threshold value for binarization. Defaults to 60.
    upperlimit : int, optional
        Upper threshold value for binarization. Defaults to 255.
    areathreshold : float, optional
        Threshold value to filter contours based on area. Defaults to 0.85.

    Returns:
    --------
    List[np.ndarray]
        List of contours.
    """
    
    if len(image.shape)>2:
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imgray = image
    
    _, thresh = cv2.threshold(imgray, lowerlimit, upperlimit, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    morph = morph.astype(np.uint8) if morph.dtype == float else morph
        
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    trend = np.quantile([cv2.contourArea(cnt) for cnt in contours],areathreshold)

    cntfilter = [cnt for cnt in contours if cv2.contourArea(cnt) >= trend]
    
    return cntfilter

def find_contours(image: np.ndarray, hull: bool = False):
    """
    Find contours in the image.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    hull : bool, optional
        Whether to find convex hull contours. Defaults to False.

    Returns:
    --------
    np.ndarray
        Contour points.
    """
    
    maskimage = image.copy()
    #imgmas = (maskimage*255).astype(np.uint8)
    ## mask must has alues btwn 0  and 255
    if np.max(maskimage)==1.:
        maskimage = maskimage * 255.
    contours = contours_from_image(maskimage)
    if hull:
        firstcontour = cv2.convexHull(contours[0])
    else:
        firstcontour = contours[0]
        
    rect = cv2.minAreaRect(firstcontour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    return box


def from_contours_to_mask(imageshape, contours, decode = True):
    
    import pycocotools.mask as mask_util
    
    masks = []
    for i in range(len(contours)):
        
        
        msk = np.zeros(imageshape, np.uint8)

        msk = cv2.drawContours(msk, [contours[i]], -1, (255,255,255), thickness=cv2.FILLED)
        
        #msk = (msk/255).astype(np.uint16)
        
        binary_mask_fortran = np.asfortranarray(msk[:,:,0])
        
        if decode:
            rle = {'counts': [], 'size': list(binary_mask_fortran.shape)}
            msk = mask_util.encode(binary_mask_fortran)
            rle["counts"] = msk["counts"].decode()
        else:
            rle = msk[:,:,0]
            
        masks.append(rle)
    
    return masks


def randomly_displace(img,maxshift = 0.15, xshift = None, yshift = None):
    """
    a Function that will randomly discplace n image given a maximum shift value

    Parameters
    -------
    img: nparray
        array data that represents the image
    maxshift: float
        maximum displacement in x and y axis

    Return
    -------
    2D array
    """
    if xshift is None:
        xoptions = list(range(-int(img.shape[0]*maxshift),int(img.shape[0]*maxshift)))
        xshift = random.choice(xoptions)

    if yshift is None:
        yoptions = list(range(-int(img.shape[1]*maxshift),int(img.shape[1]*maxshift)))
        yshift = random.choice(yoptions)

    
    if len(img.shape)>3:
        stackedimgs = []
        for t in range(img.shape[2]):
            stackedimgs.append(register_image_shift(img[:,:,t,:].copy(), [xshift,yshift]))
        stackedimgs = np.stack(stackedimgs, axis = 2)
    else:
        stackedimgs = register_image_shift(img.copy(), [xshift,yshift])

    return stackedimgs, [xshift, yshift]


def register_image_shift(data, shift):
    tform = SimilarityTransform(translation=(shift[1], shift[0]))
    
    if len(data.shape)>2:
        imglist = []
        for i in range(data.shape[2]):
            imglist.append(warp(data[:, :, i], inverse_map=tform, order=0, preserve_range=True))
        imglist = np.dstack(imglist)

    else:
        imglist = warp(data[:, :], inverse_map=tform, order=0, preserve_range=True)

    return imglist


def randomly_shiftimage(img,maxshift = 0.15):
    xoptions = list(range(-int(img.shape[0]*maxshift),int(img.shape[0]*maxshift)))
    yoptions = list(range(-int(img.shape[1]*maxshift),int(img.shape[1]*maxshift)))
            # pick angles at random
    xshift = random.choice(xoptions)
    yshift = random.choice(yoptions)
    if len(img.shape)>3:
        stackedimgs = []
        for t in range(img.shape[2]):
            stackedimgs.append(register_image_shift(img[:,:,t,:].copy(), [xshift,yshift]))
        stackedimgs = np.stack(stackedimgs, axis = 2)
    else:
        stackedimgs = register_image_shift(img.copy(), [xshift,yshift])

    return stackedimgs


def image_zoom(img: np.ndarray, zoom_factor: float = 0, 
               paddling_value: float = 0) -> np.ndarray:

    """
    Performs a centered zoom in or out on the given image without changing its dimensions. 
    The function can enlarge or shrink the image view while padding or cropping to maintain the original size.

    Parameters
    ----------
    img : np.ndarray
        The input image array.
    zoom_factor : float, optional
        The amount of zoom as a percentage [-100, 100], where positive values zoom in and negative values zoom out. 
        The default is 0, which returns the input image unaltered.
    paddling_value : Union[int, float], optional
        The value used for padding when zooming out. The default is 0.

    Returns
    -------
    np.ndarray
        The zoomed image as a numpy ndarray of the same shape as the input image.

    Raises
    ------
    AssertionError
        If the result shape does not match the input image shape.
    """
    if zoom_factor == 0:
        return img
    
    if zoom_factor>0:
        zoom_in = False
        zoom_factor = zoom_factor/100
    else:
        zoom_in = True
        zoom_factor = (-1*zoom_factor/100)
    
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    

    if zoom_in:

        y1, x1 = int(new_height)//2 , int(new_width)//2
        y2, x2 = height - y1 , width - x1
        bbox = np.array([(height//2 - y1),
                         (width//2 - x1),
                         (height//2 + y1),
                         (width//2 + x1)])

        y1, x1, y2, x2 = bbox


        cropped_img = img[y1:y2, x1:x2]

        result = cv2.resize(cropped_img, 
                        (width, height), 
                        interpolation= cv2.INTER_LINEAR
                        )
    else:
        zoom_factor = 1 if zoom_factor == 0 else zoom_factor
        
        new_height, new_width = height + int(height * zoom_factor), width + int(width * zoom_factor)
        if paddling_value == 0:
            newshape = list(img.shape)
            newshape[0] = new_height
            newshape[1] = new_width
            newimg = np.zeros(newshape).astype(img.dtype)
        else:
            newimg = np.ones(newshape).astype(img.dtype) * paddling_value
          
        pad_height1, pad_width1 = abs(new_height - height) // 2, abs(new_width - width) //2
        newimg[pad_height1:(height+pad_height1), pad_width1:(width+pad_width1)] = img
        
        result = cv2.resize(newimg, 
                        (height, width), 
                        interpolation= cv2.INTER_LINEAR
                        )
        #result = np.pad(result, pad_spec)
        
    
    assert result.shape[0] == height and result.shape[1] == width, "Resulting image shape does not match the original."

    #result[result<0.000001] = 0.0
    return result


def illumination_shift(img, valuel = []):
     

    dtype = img.dtype
    # pick illumination values randomly
    if isinstance(valuel, list):
        valuel = random.choice(valuel)
    else:
        valuel = valuel
        
    imgrgb = cv2_array_type(img)
        
    imgl,_ = shift_hsv(imgrgb, hue_shift = 0, sat_shift=0, 
                       val_shift = valuel)

    if img.shape[0] == 3:
        imgl = imgl.swapaxes(2,1).swapaxes(1,0)
    if np.max(img) < 2:
        imgl = imgl/255.
            
    return imgl.astype(dtype)

def image_rotation(img, angle = []):
            # pick angles at random

    if isinstance(angle, list):
        angle = random.choice(angle)
    else:
        angle = angle
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    transformmatrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, transformmatrix, (w, h))

    return rotated

def image_flip(img, flipcode = []):
            # pick angles at random

    if isinstance(flipcode, list):
        flipcode = random.choice(flipcode)
    else:
        flipcode = flipcode

    
    imgflipped = cv2.flip(img, 0)

    return imgflipped


def scipy_rotate(img,angle = []):
        # define some rotation angles
        
        # pick angles at random
        if isinstance(angle, list):
            angle = random.choice(angle)
        else:
            angle = angle

        # rotate volume
        rotated = ndimage.rotate(img, angle, reshape=False)
        rotated[rotated < 0] = 0

        return rotated

def start_points(size, split_size, overlap=0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(pt)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def to_standard_rgb(rgbarray: np.ndarray) -> np.ndarray:
    """
    Convert an RGB array to standard RGB using the sRGB transformation.

    Parameters
    ----------
    rgbarray : np.ndarray
        An array of RGB values with shape (..., 3). The values are expected to be in the range [0, 255].

    Returns
    -------
    np.ndarray
        An array of standard RGB values with the same shape as the input.
    Notes
    -----
    This function is based on the method described at:
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/color_space/srgb.py#L41-L75
    """
    # transform to standard rgb https://en.wikipedia.org/wiki/SRGB.
    srgb_gamma = {"A": 0.055, "PHI": 12.92, "K0": 0.04045, "GAMMA": 2.4}
    
    rgbarrayc = rgbarray.copy()
    # If the max value in the array is greater than 10, assume the values are in the range [0, 255] and normalize them to [0, 1].
    if np.max(rgbarrayc) > 10:
        rgbarrayc = rgbarrayc/255.
    
    srgb = np.where(rgbarrayc <= srgb_gamma['K0'] / srgb_gamma['PHI'], 
                        rgbarrayc * srgb_gamma['PHI'], 
                        (1 + srgb_gamma['A']) * (rgbarrayc**(1 / srgb_gamma['GAMMA'])) - srgb_gamma['A'])

    return srgb

def transformto_cielab(dataimg: np.ndarray) -> np.ndarray:
    """
    Convert RGB data to CIELAB color space.

    Parameters:
    -----------
    dataimg : np.ndarray
        Input RGB image data.

    Returns:
    --------
    np.ndarray
        CIELAB color space representation of the input RGB image data.
    """
    
    from skimage import io, color
    dataimgc = dataimg.copy()
    if dataimgc.shape[0] == 3:
        dataimgc = dataimgc.swapaxes(0,1).swapaxes(1,2)
    #dataimgc = dataimgc *255. if np.max(dataimgc ) <= 1 else dataimgc.copy()
         
    lab = color.rgb2lab(dataimgc)
    lab[lab == 0] = np.nan
    
    return lab


def transformto_hsv(dataimg: np.ndarray) -> np.ndarray:
    """
    Convert RGB data to HSV color space.

    Parameters:
    -----------
    dataimg : np.ndarray
        Input RGB image data.

    Returns:
    --------
    np.ndarray
        HSV color space representation of the input RGB image data.
    """
    
    from skimage import io, color
    dataimgc = dataimg.copy()
    if dataimgc.shape[0] == 3:
        dataimgc = dataimgc.swapaxes(0,1).swapaxes(1,2)
    dataimgc = dataimgc.copy() if np.max(dataimgc ) <= 1 else dataimgc.copy() / 255.
         
    hsv = color.rgb2hsv(dataimgc)
    hsv[hsv == 0] = np.nan
    
    return hsv


def split_image(image, nrows=None, ncols=None, overlap=0.0):

    img_height, img_width, channels = image.shape

    if nrows is None and ncols is None:
        nrows = 2
        ncols = 2

    width = math.ceil(img_width / ncols)
    height = math.ceil(img_height / nrows)

    row_off_list = start_points(img_height, height, overlap)
    col_off_list = start_points(img_width, width, overlap)
    offsets = product(col_off_list, row_off_list)

    imgtiles = []
    combs = []
    for col_off, row_off in offsets:
        imgtiles.append(image[row_off:(row_off + height), col_off:(col_off + width)])
        combs.append('{}_{}'.format(col_off,row_off))

    return imgtiles, combs


def from_array_2_jpg(arraydata,
                     ouputpath=None,
                     export_as_jpg=True,
                     size=None,
                     verbose=True):
    if ouputpath is None:
        ouputpath = "image.jpg"
        directory = ""
    else:
        directory = os.path.dirname(ouputpath)

    if arraydata.shape[0] == 3:
        arraydata = np.moveaxis(arraydata, 0, -1)

    image = Image.fromarray(arraydata.astype(np.uint8), 'RGB')
    if size is not None:
        image = image.resize(size)

    if export_as_jpg:
        Path(directory).mkdir(parents=True, exist_ok=True)

        if not ouputpath.endswith(".jpg"):
            ouputpath = ouputpath + ".jpg"

        image.save(ouputpath)

        if verbose:
            print("Image saved: {}".format(ouputpath))

    return image


def change_images_contrast(image,
                           alpha=1.0,
                           beta=0.0,
                           neg_brightness=False):
    """

    :param neg_brightness:
    :param nsamples:
    :param alpha: contrast contol value [1.0-3.0]
    :param beta: Brightness control [0-100]

    :return: list images transformed
    """
    if type(alpha) != list:
        alpha_values = [alpha]
    else:
        alpha_values = alpha

    if type(beta) != list:
        beta_values = [beta]
    else:
        beta_values = beta

    if neg_brightness:
        betshadow = []
        for i in beta_values:
            betshadow.append(i)
            betshadow.append(-1 * i)
        beta_values = betshadow

    ims_contrasted = []
    comb = []
    for alpha in alpha_values:
        for beta in beta_values:
            ims_contrasted.append(
                cv2.convertScaleAbs(image, alpha=alpha, beta=beta))

            comb.append('{}_{}'.format(alpha, beta))

    return ims_contrasted, comb


### https://scikit-image.org/docs/stable/auto_examples/filters/plot_dog.html
def diff_guassian_img(img, low_sigma = 1, high_sigma=[30,40,50,60,70]):
    """
    difference between blurred images    

    Args:
        img (_type_): _description_
        low_sigma (int, optional): _description_. Defaults to 1.
        high_sigma (list, optional): _description_. Defaults to [30,40,50,60,70].

    Returns:
        _type_: _description_
    """
    if type(high_sigma) == list:
        # pick angles at random
        high_sigma = random.choice(high_sigma)
    
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = difference_of_gaussians(img, low_sigma, high_sigma)
    else:
        imgnew = np.zeros(img.shape)
        for i in range(img.shape[0]):
            imgnew[i] = difference_of_gaussians(img[i], low_sigma, high_sigma)
        img = imgnew
        
    return img, [str(high_sigma)]


## taken from https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py
#### Histogram Equalization and Adaptive Histogram Equalization
def clahe_img(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    params:
     clip_limit: This is the threshold for contrast limiting
     tile_grid_size: Divides the input image into M x N tiles and then applies histogram equalization to each local tile
    """
    if type(clip_limit) == list:
        # pick angles at random
        clip_limit = random.choice(clip_limit)

    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img, [str(clip_limit)]

### hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
def shift_hsv(img, hue_shift, sat_shift, val_shift):

    if type(sat_shift) == list:
        # pick angles at random
        sat_shift = random.choice(sat_shift)

    if type(hue_shift) == list:
        # pick angles at random
        hue_shift = random.choice(hue_shift)

    if type(val_shift) == list:
        # pick angles at random
        val_shift = random.choice(val_shift)

    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.uint8)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    if sat_shift != 0:
        lut_sat = np.arange(0, 256, dtype=np.uint8)
        lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)

    if val_shift != 0:
        lut_val = np.arange(0, 256, dtype=np.uint8)
        lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
        if val_shift>0:
            lut_val[np.where(lut_val == 255)[0][0]:] = 255
        val = cv2.LUT(val, lut_val)

    img = np.stack((hue, sat, val), axis = 2).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img, ['{}_{}_{}'.format(hue_shift, sat_shift, val_shift)]
## rotate image


def rotate_npimage(image, angle=[0]):

    if type(angle) == list:
        # pick angles at random
        angle = random.choice(angle)

    pil_image = Image.fromarray(image)

    img = pil_image.rotate(angle)

    return np.array(img), [str(angle)]


def rotate_xyxoords(x, y, anglerad, imgsize, xypercentage=True):
    center_x = imgsize[1] / 2
    center_y = imgsize[0] / 2

    xp = ((x - center_x) * cos(anglerad) - (y - center_y) * sin(anglerad) + center_x)
    yp = ((x - center_x) * sin(anglerad) + (y - center_y) * cos(anglerad) + center_y)

    if imgsize[0] != 0:
        if xp > imgsize[1]:
            xp = imgsize[1]
        if yp > imgsize[0]:
            yp = imgsize[0]

    if xypercentage:
        xp, yp = xp / imgsize[1], yp / imgsize[0]

    return xp, yp


def rotate_yolobb(yolo_bb, imgsize, angle):
    angclock = -1 * angle

    xc = float(yolo_bb[1]) * imgsize[1]
    yc = float(yolo_bb[2]) * imgsize[0]
    xr, yr = rotate_xyxoords(xc, yc, radians(angclock), imgsize)
    w_orig = yolo_bb[3]
    h_orig = yolo_bb[4]
    wr = np.abs(sin(radians(angclock))) * h_orig + np.abs(cos(radians(angclock)) * w_orig)
    hr = np.abs(cos(radians(angclock))) * h_orig + np.abs(sin(radians(angclock)) * w_orig)

    # l, r, t, b = from_yolo_toxy(origimgbb, (imgorig.shape[1],imgorig.shape[0]))
    # coords1 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords2 = rotate_xyxoords(r,b,radians(angclock),rotatedimg.shape)
    # coords3 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords4 = rotate_xyxoords(l,t,radians(angclock),rotatedimg.shape)
    # w = math.sqrt(math.pow((coords1[0] - coords2[0]),2)+math.pow((coords1[1] - coords2[1]),2))
    # h = math.sqrt(math.pow((coords3[0] - coords4[0]),2)+math.pow((coords3[1] - coords4[1]),2))
    return [yolo_bb[0], xr, yr, wr, hr]


### expand


def expand_npimage(image, ratio=25, keep_size=True):
    if type(ratio) == list:
        # pick angles at random
        ratio = random.choice(ratio)

    pil_image = Image.fromarray(image)
    width = int(pil_image.size[0] * ratio / 100)
    height = int(pil_image.size[1] * ratio / 100)
    st = ImageOps.expand(pil_image, border=(width, height), fill='white')

    if keep_size:
        st = resize_npimage(np.array(st), image.shape)

    return np.array(st), [str(ratio)]


## blur image


def blur_image(image, radius=[0]):

    if type(radius) == list:
        # pick angles at random
        radius = random.choice(radius)

    pil_image = Image.fromarray(image)
    img = pil_image.filter(ImageFilter.GaussianBlur(radius))

    return np.array(img), [str(radius)]



def cartimg_topolar_transform(nparray, anglestep = 5, max_angle = 360, expand_ratio = 40, nathreshhold = 5):

    xsize = nparray.shape[1]
    ysize = nparray.shape[0]
    
    if expand_ratio is None:
        mayoraxisref = [xsize,ysize] if xsize > ysize else [ysize,xsize]
        expand_ratio = (mayoraxisref[0]/mayoraxisref[1] - 1)*100

    newwidth = int(xsize * expand_ratio / 100)
    newheight = int(ysize * expand_ratio / 100)

    # exand the image for not having problem whn one axis is bigger than other
    pil_imgeexpand = ImageOps.expand(Image.fromarray(nparray), 
                                     border=(newwidth, newheight), fill=np.nan)

    
    listacrossvalues = []
    distances = []
    # the image will be rotated, then the vertical values were be extracted with each new angle
    for angle in range(0, max_angle, anglestep):
        
        imgrotated = pil_imgeexpand.copy().rotate(angle)
        imgarray = np.array(imgrotated)
        cpointy = int(imgarray.shape[0]/2)
        cpointx = int(imgarray.shape[1]/2)

        valuesacrossy = []
        
        i=(cpointy+0)
        coordsacrossy = []
        # it is important to have non values as nan, if there are to many non values in a row it will stop
        countna = 0 
        while (countna<nathreshhold) and (i<(imgarray.shape[0]-1)):
            
            if np.isnan(imgarray[i,cpointx]):
                countna+=1
            else:
                coordsacrossy.append(i- cpointy)
                valuesacrossy.append(imgarray[i,cpointx])
                countna = 0
            i+=1

        distances.append(coordsacrossy)
        listacrossvalues.append(valuesacrossy)
    
    maxval = 0
    nrowid =0 
    for i in range(len(distances)):
        if maxval < len(distances[i]):
            maxval = len(distances[i])
            
            nrowid = distances[i][len(distances[i])-1]
    
    for i in range(len(listacrossvalues)):
        listacrossvalues[i] = listacrossvalues[i] + [np.nan for j in range((nrowid+1) - len(listacrossvalues[i]))]
    

    return [distances, np.array(listacrossvalues)]



####



# https://en.wikipedia.org/wiki/Histogram_equalization
def hist_equalization(np2dimg):
    if 'f' in np2dimg.dtype.str:
        np2dimg[np.isnan(np2dimg)] = 0
        np2dimg = np2dimg.astype('uint8')

    hist,_ = np.histogram(np2dimg.flatten(),256,[0,256])
    cdf = hist.cumsum()
    #cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[np2dimg]
    
def hist_3dimg(np3dimg):
    ref = np.min(np3dimg.shape)
    mind = [i for i in range(len(np3dimg.shape)) if np3dimg.shape[i] == ref]
    imgeqlist =[]
    for i in range(np3dimg.shape[mind[0]]):
        
        imgeqlist.append(hist_equalization(np3dimg[i]))
    
    return np.array(imgeqlist)

def change_bordersvaluesasna(nptwodarray, bufferna):
    intborderx = int(nptwodarray.shape[0]*(bufferna/100))
    intbordery = int(nptwodarray.shape[1]*(bufferna/100))
    nptwodarray[
        (nptwodarray.shape[0]-intborderx):nptwodarray.shape[0],
            :] = np.nan
    nptwodarray[:,
            (nptwodarray.shape[1]-intbordery):nptwodarray.shape[1]] = np.nan
    nptwodarray[:,
            0:intbordery] = np.nan
    nptwodarray[0:intborderx,:] = np.nan

    return nptwodarray

def getcenter_from_hull(npgrayimage, buffernaprc = 15):
    nonantmpimg = npgrayimage.copy()
    if buffernaprc is not None:
        nonantmpimg = change_bordersvaluesasna(nonantmpimg, bufferna=buffernaprc)

    nonantmpimg[np.isnan(nonantmpimg)] = 0

    coords = np.transpose(np.nonzero(nonantmpimg))
    hull = ConvexHull(coords)

    cx = np.mean(hull.points[hull.vertices,0])
    cy = np.mean(hull.points[hull.vertices,1])

    return int(cx),int(cy)


def border_distance_fromgrayimg(grimg):
    contours, _ = cv2.findContours(grimg, 
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
                                  
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly)

    centers = centers[np.where(radius == np.max(radius))[0][0]]
    radius = radius[np.where(radius == np.max(radius))[0][0]]

    return centers,radius

def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype('float'), axis=2)
    im2_gray = np.sum(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')


def phase_convolution(refdata, targetdata):
    corr_img = cross_image(refdata,
                           targetdata)
    shape = corr_img.shape

    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    maxima = np.unravel_index(np.argmax(corr_img), shape)
    shifts = np.array(maxima, dtype=np.float64)

    shifts = np.array(shifts) - midpoints

    return shifts

def radial_filter(nparray, anglestep = 5, max_angle = 360, nathreshhold = 5, nanvalue = None):
    """ Function to filter those pixels that that are not in contarct with the main image

    Args:
        nparray (_type_): image numpy array 2D
        anglestep (int, optional): anlge used to go though the figure. Defaults to 5.
        max_angle (int, optional): up to what angle. Defaults to 360.
        nathreshhold (int, optional): maximum separation. Defaults to 5.
        nanvalue (_type_, optional): if nan value in the image is represented by a specfic value. Defaults to None.

    Returns:
        _type_: image numpy array 2D
    """
    origimg = nparray.copy()
    if nanvalue is not None:
        nanmask = nparray == nanvalue
        origimg[nanmask] = np.nan
    center_x = int(nparray.shape[1]/2)
    center_y = int(nparray.shape[0]/2)

    
    if np.isnan(origimg[center_y,center_x]):
        center_y, center_x = getcenter_from_hull(origimg)

    if np.isnan(origimg[center_y,center_x]):
        warnings.warn("there are no data in the image center")
        origimg[center_y,center_x] = 0

    modimg = np.empty(nparray.shape)
    modimg[:] = np.nan

    #mxanglerad = radians(360)
    for angle in range(0,max_angle,anglestep):

        anglerad = radians(angle)
        xp = ((center_x*2 - center_x) * cos(anglerad) - (center_y*2 - center_y) * sin(anglerad) + center_x)
        yp = ((center_x*2 - center_x) * sin(anglerad) + (center_y*2 - center_y) * cos(anglerad) + center_y)

        x1, y1 = center_x, center_y
        x2, y2 = xp,yp
        rr, cc, _ = line_aa(y1, x1, int(y2),int(x2))
        countna = 0 
        for i,j in zip(rr,cc):
            #print(i,j,m)
            if i < (nparray.shape[0]) and j < (nparray.shape[1]) and j >= 0 and i >=0:
                         
                if countna>=nathreshhold:
                    #modimg[i,j] = np.nan
                    break
                try:
                    if np.isnan(origimg[i,j]):
                        countna+=1
                    else:
                        modimg[i,j] = origimg[i,j]
                        countna = 0
                except:
                    break
            else:
                break
    
    if nanvalue is not None:
        modimg[np.isnan(modimg)] = nanvalue
        
    return modimg

    

# create filters by contour

def remove_smallpixels(img,imgrefpos = 0,thresh = 0.7):
    channelimg = img[imgrefpos].copy().astype(np.uint8)
    channelimg[channelimg<0] = 0
    channelimg[channelimg>0] = 255
    ## get mask
    smallpixelsmask = calculate_mask_contourarea(channelimg, threhold = thresh)
    
    newimgs = np.zeros(img.shape, dtype=img.dtype)
    
    for i in range(img.shape[0]):
        newimgs[i] = cv2.bitwise_and(img[i], img[i], mask=smallpixelsmask)
    
    return newimgs
def calculate_mask_contourarea(image,threhold = 0.7):
    ret, thresh = cv2.threshold(image, 127, 255, 0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    allarea = mask.shape[0] * mask.shape[0]
    cntmaxarea = np.max([cv2.contourArea(cnt) for cnt in contours])

    for i, cnt in enumerate(contours):
        #if hierarchy[0][i][2] != -1:
        if (cv2.contourArea(cnt)/allarea)/(cntmaxarea/allarea) >threhold:
            cv2.drawContours(mask, [cnt], 0, (255), -1)
        
    return mask
    