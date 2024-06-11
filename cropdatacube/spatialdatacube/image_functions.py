
import numpy as np
import random
from scipy import ndimage

from PIL import Image, ImageOps
import cv2 as cv



###


def calculate_differencesinshape(height,width, distance):
    dif_height = (distance-height//2) if distance > height//2 else 0
    dif_width = (distance-width//2) if distance > width//2 else 0
    
    return dif_height, dif_width

def img_padding(img, distancefromcenter):
    distancefromcenter = int(distancefromcenter)
    height, width = img.shape[0], img.shape[1]

    dif_height, dif_width = calculate_differencesinshape(height,width, distancefromcenter)

    newimg = np.zeros((img.shape[0]+dif_height*2, img.shape[1]+dif_width*2, img.shape[2]))
    newimg[dif_height:height+dif_height,dif_width:width+dif_width] = img

    return newimg

def border_distance_fromgrayimg(grimg):
    contours, _ = cv.findContours(grimg, 
                                  cv.RETR_TREE,
                                  cv.CHAIN_APPROX_SIMPLE)
                                  
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly)

    centers = centers[np.where(radius == np.max(radius))[0][0]]
    radius = radius[np.where(radius == np.max(radius))[0][0]]

    return centers,radius



## data augmentation

#https://keras.io/examples/vision/3D_image_classification/

#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions

import cv2

def cv2_clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height), 
                        interpolation= cv2.INTER_LINEAR
                        )
    result = np.pad(result, pad_spec)
    assert result.shape[0] == height and result.shape[1] == width
    result[result<0.000001] = 0.0
    return result


def scipy_rotate(volume,angles = [-330,-225,-180,-90, -45, -15, 15, 45, 90,180,225, 330]):
        # define some rotation angles
        
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

def clipped_zoom(img, zoom_factor, **kwargs): 
    h, w = img.shape[:2] # For multichannel images we don't want to apply the zoom factor to the RGB # dimension, so instead we create a tuple of zoom factors, one per array 
    # dimension, with 1's for any trailing dimensions after the width and height. 
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2) # Zooming out 
    if zoom_factor < 1: # Bounding box of the zoomed-out image within the output array 
        zh = int(np.round(h * zoom_factor)) 
        zw = int(np.round(w * zoom_factor)) 
        top = (h - zh) // 2 
        left = (w - zw) // 2 # Zero-padding 
        out = np.zeros_like(img) 
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs) # Zooming in 
    elif zoom_factor > 1: # Bounding box of the zoomed-in region within the input array 
        zh = int(np.round(h / zoom_factor)) 
        zw = int(np.round(w / zoom_factor)) 
        top = (h - zh) // 2 
        left = (w - zw) // 2 
        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs) # `out` might still be slightly larger than `img` due to rounding, so
         # trim off any extra pixels at the edges 
        
        trim_top = ((out.shape[0] - h) // 2) 
        trim_left = ((out.shape[1] - w) // 2) 
        out = out[trim_top:trim_top+h, trim_left:trim_left+w] # 
    else: 
        out = img 
    
    return out 

def randomly_zoom(volume):
        # define some rotation angles
        zooms = [1.75,1.5,1.25, 0.75, 0.85]
        # pick angles at random
        z = random.choice(zooms)
        
        # rotate volume
        volume = clipped_zoom(volume, z)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

def randomly_cv2_zoom(mltimage):
        # define some rotation angles
        zooms = [1.5,1.25, 0.75, 0.85]
        # pick angles at random
        z = random.choice(zooms)
        
        # zoom image
        # HWDC
        stackedimgs = []
        if len(mltimage.shape)>3:
            for t in range(mltimage.shape[2]):
                stackedimgs.append(cv2_clipped_zoom(mltimage[:,:,t,:].copy(), z))
            stackedimgs = np.stack(stackedimgs, axis = 2)
        else:
            stackedimgs = cv2_clipped_zoom(mltimage.copy(), z)

        return stackedimgs



def tranform_z2dinto_profiles(z2dimage, axis = 'x',
                              zmin = 0, zmax = 60,cms = 1, 
                              scalez = 10, slices_step = 1, initval = 1, **kwargs
                             ):
    
    """
    a functions to reshape the [x y z] data to a [x z chanel] image

    ----------
    Parameters
    data : numpy 2d array
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values
    axis :int, optional
        which axis will be transformed along with z [ 'x' , 'y' ]
    
    zmin : int, optional
        a value that will be taken as minimun for z axis
    zmax : int, optional
        a value that will be taken as maximun for z axis

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

def from_to2d_zarray(zvalues, 
                     fill_values = None,
                     axis = 'x', 
                     zmin = 0, zmax = 60,cms = 1, scalez = 10, slices_step = 1, initval = 1, **kwargs):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    data : numpy 2d array
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values
    axis :int, optional
        which axis will be transformed along with z [ 'x' , 'y' ]
    
    zmin : int, optional
        a value that will be taken as minimun for z axis
    zmax : int, optional
        a value that will be taken as maximun for z axis

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if axis=='x':
        lenaxis = zvalues.shape[0]
        slicedstr = 'zvalues[j]'
        if fill_values is not None:
            sentvalues = '[data[j]]'
        else:
            sentvalues = 'fill_values'
    else:
        lenaxis = zvalues.shape[1]
        slicedstr = 'zvalues[:,j]'
        if fill_values is not None:
            sentvalues = '[data[:,j]]'
        else:
            sentvalues = 'fill_values'


    zimg = np.array([i for i in range(zmin*scalez, zmax*scalez, int(float(cms) * float(scalez)))])
    

    slicelist = []
    for j in range(initval*slices_step,(lenaxis-(slices_step)),slices_step):
        slicedataz = eval(slicedstr)
        if np.sum(np.logical_not(np.isnan(slicedataz))) > 0:
            slicedata = eval(sentvalues)
            z2dimg = singlexy_to2d_zarray(
                                           
                                          slicedataz, 
                                          fill_Values = slicedata,
                                          scalez = scalez, 
                                          referencearray = zimg,**kwargs)[0]
        else:
            z2dimg = np.zeros((len(zimg),len(slicedataz)))
        slicelist.append(z2dimg)

    return np.array(slicelist)    


#@ray.remote
def transform_to_heightprofile(
                     z2img, 
                     fill_values = None,
                     slices_step = 6,
                     axis = 0, 
                     zmax_cm = 60,
                     scalez = 1, 
                     
                     nslices = None,
                     initial_slice = 1, **kwargs):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    zvalues: numpy 2d array
        an array that contains the z values
    fill_values : numpy 2d array, optional
        an array that contains the chanel data to be transformed as x z axis1
    axis :int, optional
        which axis will be transformed along with z [ 0 , 1 ]
    
    scalez : int, optional
        a value that will be taken as minimun for scaling z
    zmax_cm : int, optional
        a value that will be taken as maximun for z axis in cm

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if axis == 0:
        datatotransform = z2img
    else:
        datatotransform = z2img.swapaxes(1,0)
        if fill_values is not None:
            fill_values = fill_values.swapaxes(1,0)

    lenaxis = datatotransform.shape[0]
    newheight = int(zmax_cm*scalez)
    if nslices is not None:
        slices_step = int(lenaxis/nslices)
        print(slices_step)
        
    slicelist = []
    for j in range(initial_slice*slices_step,
                    (lenaxis-(slices_step)),slices_step):
        slicedataz = datatotransform[j]
        if np.sum(np.logical_not(np.isnan(slicedataz))) > 0:
            z2dimg = singleline_to2d_slice(datatotransform[j], 
                                            fill_Values = datatotransform[j],
                                            scalez = scalez, 
                                            referenheight= newheight,
                                            **kwargs
                                            )

        else:
            z2dimg = np.zeros((newheight,len(slicedataz)))
        slicelist.append(z2dimg)

    return np.array(slicelist) 



def singleline_to2d_slice(zvalues, fill_Values = None, scalez = 1, 
                         referenheight = None, 
                         barstyle = True, flip =True, fliplefttoright = False):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    fill_Values : numpy 2d array, optional
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values

    scalez: floar, optional
        a numerical value that will be used to scale height images
    
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if fill_Values is None:
        barstyle = False
    else:
        barstyle = True
        
    npzeros = np.zeros((len(zvalues),referenheight))
    for xi in range(len(zvalues)):
        xivals = zvalues[xi]
        if not np.isnan(zvalues[xi]):
            zpos = (np.round(xivals,1)*scalez).astype(int)
            zpos = zpos - zpos%scalez
            
            if barstyle:
                npzeros[xi,:int((zpos))] = fill_Values[xi]
            else:
                npzeros[xi,:int((zpos))] = 1

    if flip:
        npzeros = npzeros.swapaxes(0,1)
        npzeros = Image.fromarray(npzeros)
        npzeros = np.array(ImageOps.flip(npzeros))
    if fliplefttoright:
        npzeros = np.array(Image.fromarray(npzeros).transpose(Image.FLIP_LEFT_RIGHT))
    
    return npzeros

