
import numpy as np
from math import cos, sin, radians
import os

import cv2
import copy

import inspect
from typing import List, Dict

def xyxy_to_xywh(bbs):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = [0]*len(bbs)
    y[0] = (bbs[0] + bbs[2]) / 2  # x center
    y[1] = (bbs[1] + bbs[3]) / 2  # y center
    y[2] = abs(bbs[2] - bbs[0])  # width
    y[3] = abs(bbs[3] - bbs[1])  # height
    return y

def xywh_to_xyxy(yolo_style, size):
    x, y, w, h = yolo_style

    l = int((x - w / 2))
    r = int((x + w / 2))
    t = int((y - h / 2))
    b = int((y + h / 2))

    if l < 0:
        l = 0

    if t < 0:
        t = 0

    return (l, r, t, b)

def flip_bb(bbs, img_width: int, img_height: int, axis_order: List[int] = [0, 2]) -> np.ndarray:
    """
    Flip bounding boxes along the specified axes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    axis_order : List[int], optional
        Order of axes for flipping. Default is [0, 2].

    Returns
    -------
    np.ndarray
        Flipped bounding boxes.
    """
    img_center = np.array((img_width/2,img_height/2))
    img_center = np.hstack((img_center, img_center))

    bbs[:, axis_order] += 2*(img_center[axis_order] - bbs[:, axis_order])

    box_ = abs(bbs[:, axis_order[0]] - bbs[:, axis_order[1]])

    bbs[:, axis_order[0]] -= box_
    bbs[:, axis_order[1]] += box_

    return bbs

def flip_bb_using_cv2flag(bbs: np.ndarray, img_width: int, img_height: int, flip_code: int) -> np.ndarray:
    """
    Flip bounding boxes using OpenCV flip code.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    flip_code : int
        Flip code (0, 1, or -1) as used in OpenCV.
        The flip code defining the flip direction:
        - 0 means flipping around the x-axis (vertical flip).
        - Positive value (1) means flipping around the y-axis (horizontal flip).
        - Negative value (-1) means flipping around both axes (vertical and horizontal flip).

    Returns
    -------
    np.ndarray
        Flipped bounding boxes.
    """
    
    bbtr = copy.deepcopy(bbs).astype(float)
    
    if flip_code == 0:
        return flip_bb(bbtr,img_width, img_height, [1,3])
    if flip_code == 1:
        return flip_bb(bbtr,img_width, img_height, [0,2])
    if flip_code == -1:
        for fliporder in [[0,2],[1,3]]:
            bbtr = flip_bb(bbtr,img_width, img_height, fliporder)
            
        return bbtr
        
    else:
        raise ValueError(f'there is no implementation for {flip_code}')


def get_corners(bboxes: np.ndarray) -> np.ndarray:
    """
    Get the corners of bounding boxes.

    Parameters
    ----------
    bboxes : np.ndarray
        Array of bounding boxes.

    Returns
    -------
    np.ndarray
        Array of corners for each bounding box.
    """

    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners: np.ndarray, angle: float, cx: float, cy: float) -> np.ndarray:
    """
    Rotate the corners of bounding boxes around a center.

    Parameters
    ----------
    corners : np.ndarray
        Array of corners for each bounding box.
    angle : float
        Rotation angle in degrees.
    cx : float
        X-coordinate of the center of rotation.
    cy : float
        Y-coordinate of the center of rotation.

    Returns
    -------
    np.ndarray
        Rotated corners.
    """
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    calculated = warp_corners(corners,  M)
    return calculated

def get_enclosing_box(corners: np.ndarray) -> np.ndarray:
    """
    Get the enclosing box for the given corners.

    Parameters
    ----------
    corners : np.ndarray
        Array of corners for each bounding box.

    Returns
    -------
    np.ndarray
        Enclosing bounding boxes.
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final


def warp_corners(corners: np.ndarray, transform_affine: np.ndarray) -> np.ndarray:
    """
    Apply an affine transformation to the corners of bounding boxes.

    Parameters
    ----------
    corners : np.ndarray
        Array of corners for each bounding box.
    transform_affine : np.ndarray
        Affine transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed corners.
    """
    
    corners = corners.reshape(-1,2)

    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    calculated = np.dot(transform_affine,corners.T).T

    return calculated.reshape(-1,8)
    
    

def bbs_rotation(bbs: np.ndarray, img_width: int, img_height: int, angle: float) -> np.ndarray:
    """
    Rotate bounding boxes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated bounding boxes.
    """
    cx = img_width/2
    cy = img_height/2
    
    corners = get_corners(bbs)

    rotated_bbs = get_enclosing_box(rotate_box(corners[:,:8], angle, cx, cy))

    return rotated_bbs

def shear_bbs(bbs: np.ndarray, img_width: int, img_height: int, shear_x: float, shear_y: float) -> np.ndarray:
    """
    Shear bounding boxes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    shear_x : float
        Shear factor along the x-axis.
    shear_y : float
        Shear factor along the y-axis.

    Returns
    -------
    np.ndarray
        Sheared bounding boxes.
    """
    transformmatrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    transformmatrix[0,2] = -transformmatrix[0,1] * img_width/2
    transformmatrix[1,2] = -transformmatrix[1,0] * img_height/2
    bbscopy = bbs.copy()
    corners = get_corners(bbscopy)

    calculated = warp_corners(corners,  transformmatrix)
    transformed_bbs = get_enclosing_box(calculated.reshape(-1,8))
    
    return transformed_bbs



def expand_bb(bbs: np.ndarray, img_width: int, img_height: int, zoom_factor: float) -> np.ndarray:
    """
    Expand bounding boxes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    zoom_factor : float
        Zoom factor for expanding bounding boxes.

    Returns
    -------
    np.ndarray
        Expanded bounding boxes.
    """

    
    wbbs = (bbs[:,2] - bbs[:,0] )
    hbbs = (bbs[:,3] - bbs[:,1] )
    
    cx = ((bbs[:,0] + bbs[:,2] )/2)*(1+zoom_factor) - ((img_width*zoom_factor)/2)
    cy = ((bbs[:,1] + bbs[:,3] )/2)*(1+zoom_factor) - ((img_height*zoom_factor)/2)

    x1 = cx - (wbbs*(1+zoom_factor))/2
    x2 = cx + (wbbs*(1+zoom_factor))/2
    y1 = cy - (hbbs*(1+zoom_factor))/2
    y2 = cy + (hbbs*(1+zoom_factor))/2

    bewbbs = np.dstack([x1,y1,x2,y2])[0]
    
    return bewbbs

def translate_bbs(bbs: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """
    Translate bounding boxes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    shift_x : float
        Shift along the x-axis.
    shift_y : float
        Shift along the y-axis.

    Returns
    -------
    np.ndarray
        Translated bounding boxes.
    """
    
    transformmatrix = np.float32([[1, 0, shift_y], 
                                  [0, 1, shift_x]])
    
    transformmatrix[0,2] = -(transformmatrix[0,2]) #* w/2
    transformmatrix[1,2] = -(transformmatrix[1,2]) 
    
    bbscopy = bbs.copy()
    corners = get_corners(bbscopy)

    calculated = warp_corners(corners,  transformmatrix)

    translated_bbs = get_enclosing_box(calculated.reshape(-1,8))
    
    
    return translated_bbs
    

def perspective_bbs(bbs: np.ndarray, perspective_x: float, perspective_y: float) -> np.ndarray:
    """
    Apply a perspective transformation to bounding boxes.

    Parameters
    ----------
    bbs : np.ndarray
        Array of bounding boxes.
    perspective_x : float
        Perspective transformation factor along the x-axis.
    perspective_y : float
        Perspective transformation factor along the y-axis.

    Returns
    -------
    np.ndarray
        Transformed bounding boxes.
    """
    ## modified from yolov5
    transformmatrix = np.eye(3)
    transformmatrix[2,0] = perspective_y
    transformmatrix[2,1] = perspective_x

    bbscopy = bbs.copy()
    corners = get_corners(bbscopy)
    xy = np.ones((corners.shape[0] * 4, 3))
    xy[:, :2] = corners.reshape(corners.shape[0] * 4, 2)
    
    xy = xy @ transformmatrix.T
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(corners.shape[0], 8)

    transformed_bbs = get_enclosing_box(xy)
    
    return transformed_bbs

class BBAugmentation():
    """
    Class for performing various bounding box augmentations.
    """
    
    @property
    def _available_options(self):
        """
        Available augmentation options.

        Returns
        -------
        Dict[str, callable]
            Dictionary of augmentation functions.
        """
        
        return {'rotation': bbs_rotation,
         'flip': flip_bb_using_cv2flag,
         'zoom': expand_bb,
         'shear': shear_bbs,
         'shift': translate_bbs,
         'perspective': perspective_bbs}
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _match_with_image_size(bbs: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Match bounding boxes with image size, ensuring they are within image bounds.

        Parameters
        ----------
        bbs : np.ndarray
            Array of bounding boxes.
        img_width : int
            Width of the image.
        img_height : int
            Height of the image.

        Returns
        -------
        np.ndarray
            Bounding boxes matched to image size.
        """
        clip_box = [0,0,img_width, img_height]
        bbox = np.ones(bbs.shape).astype(bbs.dtype)
        #x_min = np.maximum(bbs[:,0], clip_box[0]).reshape(-1,1)
        #y_min = np.maximum(bbs[:,1], clip_box[1]).reshape(-1,1)
        #x_max = np.minimum(bbs[:,2], clip_box[2]).reshape(-1,1)
        #y_max = np.minimum(bbs[:,3], clip_box[3]).reshape(-1,1)
        bbox[:,[0,2]] = bbs[:,[0,2]].clip(0,img_width)
        bbox[:,[1,3]] = bbs[:,[1,3]].clip(0,img_height)
        
        #bbox = np.hstack((x_min, y_min, x_max, y_max))
        return bbox
    
    def _checkaugfun(self, aug_fun: str) -> bool:
        """
        Check if the augmentation function is available.

        Parameters
        ----------
        aug_fun : str
            Augmentation function name.

        Returns
        -------
        bool
            True if the function is available, False otherwise.
        """
        current_options =list(self._available_options.keys())
        #assert aug_fun in current_options, f"{aug_fun} is not implemented, current options {current_options}"
        return aug_fun in current_options
    
    def chain_transformation(self, bb: np.ndarray, chain_parameters: Dict[str, List], img_width: int = None, img_height: int = None) -> np.ndarray:
        """
        Apply a chain of transformations to bounding boxes.

        Parameters
        ----------
        bb : np.ndarray
            Array of bounding boxes.
        chain_parameters : Dict[str, List]
            Dictionary of transformation parameters.
        img_width : int, optional
            Width of the image.
        img_height : int, optional
            Height of the image.

        Returns
        -------
        np.ndarray
            Transformed bounding boxes.
        """
        bb_tr = bb.copy()
        
        for k, v in chain_parameters.items():
            v = v if isinstance(v, list) else [v]
            
            bb_tr = self.transform(bb_tr,k, img_width, img_height, 
                                    *v)
        
        return bb_tr
    
    @staticmethod
    def filter_bbs(tr_bbs: np.ndarray, orig_bbs: np.ndarray, upper_area_thr: float = 4, min_area_thr: float = 0.15, aspectratio_thr: float = 8) -> np.ndarray:
        """
        Filter bounding boxes based on area and aspect ratio.

        Parameters
        ----------
        tr_bbs : np.ndarray
            Transformed bounding boxes.
        orig_bbs : np.ndarray
            Original bounding boxes.
        upper_area_thr : float, optional
            Upper area threshold for filtering. Default is 4.
        min_area_thr : float, optional
            Minimum area threshold for filtering. Default is 0.15.
        aspectratio_thr : float, optional
            Aspect ratio threshold for filtering. Default is 100.

        Returns
        -------
        np.ndarray
            Boolean mask indicating valid bounding boxes.
        """
        ## modified from yolov5
        divfactor = 1e-15
        w2 = tr_bbs[:,2] - tr_bbs[:,0]
        h2 = tr_bbs[:,3] - tr_bbs[:,1]
        w1 = orig_bbs[:,2] - orig_bbs[:,0]
        h1 = orig_bbs[:,3] - orig_bbs[:,1]
        ar = np.maximum(w2 / (h2 + divfactor), h2 / (w2 + divfactor)) 

        return (w2 * h2 / (w1 * h1 + divfactor) > min_area_thr) & (w2 * h2 / (w1 * h1 + divfactor) < upper_area_thr) & (ar < aspectratio_thr)
        
        
        
    def transform(self, bb: np.ndarray, aug_fun: str, img_width: int = None, img_height: int = None, *args) -> np.ndarray:
        """
        Apply a single transformation to bounding boxes.

        Parameters
        ----------
        bb : np.ndarray
            Array of bounding boxes.
        aug_fun : str
            Augmentation function name. Available 'rotation', 'flip', 'zoom', 'shear', 'shift','perspective'
        img_width : int, optional
            Width of the image.
        img_height : int, optional
            Height of the image.
        args : tuple
            Additional arguments for the augmentation function.

        Returns
        -------
        np.ndarray
            Transformed bounding boxes.
        """
        
        if not self._checkaugfun(aug_fun):
            return bb
        transform_fun = self._available_options[aug_fun]
        params = list(inspect.signature(transform_fun).parameters)
        
        if 'img_height' in  params:
            transformed_bb = transform_fun(bb,img_width, img_height, *args)
        else:
            transformed_bb = transform_fun(bb, *args)

        transformed_bb = self._match_with_image_size(transformed_bb, img_width, img_height)
            
        return transformed_bb
        

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

def rotate_yolobb(yolobb,imageshape, angle):
    angclock = -1 * angle
    
    xc = float(yolobb[1]) * imageshape[1]
    yc = float(yolobb[2]) * imageshape[0]
    xr, yr = rotate_xyxoords(xc, yc, radians(angclock), imageshape)
    w_orig = yolobb[3]
    h_orig = yolobb[4]
    wr = np.abs(sin(radians(angclock))) * h_orig + np.abs(cos(radians(angclock)) * w_orig)
    hr = np.abs(cos(radians(angclock))) * h_orig + np.abs(sin(radians(angclock)) * w_orig)

    # l, r, t, b = from_yolo_toxy(origimgbb, (imgorig.shape[1],imgorig.shape[0]))
    # coords1 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords2 = rotate_xyxoords(r,b,radians(angclock),rotatedimg.shape)
    # coords3 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords4 = rotate_xyxoords(l,t,radians(angclock),rotatedimg.shape)
    # w = math.sqrt(math.pow((coords1[0] - coords2[0]),2)+math.pow((coords1[1] - coords2[1]),2))
    # h = math.sqrt(math.pow((coords3[0] - coords4[0]),2)+math.pow((coords3[1] - coords4[1]),2))
    return [yolobb[0], xr, yr, wr, hr]




def label_transform(imageshape, yolobb, augtype, combination, nrep = 1):
    
    if augtype == 'expand':
        attrs = float(combination[0])
        newbb = calculate_expanded_label( yolobb, imageshape,ratio = attrs)

    if augtype =='clahe_img':
        newbb = yolobb

    if augtype == 'hsv':
        newbb = yolobb
        
    if augtype == 'contrast':
        newbb = yolobb
    
    if augtype == 'blur':
        newbb = yolobb
    
    if augtype == 'rotate':
        attrs = float(combination[0])
        newbb = []
        for yolobbsingle in yolobb:
            newbb.append(rotate_yolobb(yolobbsingle, imageshape,angle = attrs))

    if nrep>1:
        newbb = [newbb for i in range(nrep)]
        
    return newbb


def save_yololabels(bbyolo, fn,outputdir = None):

    if outputdir is not None:
        fn = os.path.join(outputdir, fn)
    if bbyolo is not None:
        with open(fn, 'w') as dst:
            for i in range(len(bbyolo)):
                strlist = [str(int(bbyolo[i][0]))]
                for j in range(1,len(bbyolo[i])):
                    strlist.append(str(bbyolo[i][j]))
                if len(bbyolo)-1 == i:
                    dst.writelines(" ".join(strlist))
                else:
                    dst.writelines(" ".join(strlist) + '\n')


def from_yolo_toxy(yolo_style, size):
    dh, dw = size
    _, x, y, w, h = yolo_style

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    return (l, r, t, b)


def percentage_to_bb(bb, size):
    ymin = int(bb[0] * size[1])  # xmin
    xmin = int(bb[1] * size[0])  # ymin
    ymax = int(bb[2] * size[1])  # xmax
    xmax = int(bb[3] * size[0])  # ymax

    return np.array([[xmin, ymin, xmax, ymax]])


def bb_topercentage(bb, size):
    xmin = bb[0] / size[1]  # xmin
    ymin = bb[1] / size[0]  # ymin
    xmax = bb[2] / size[1]  # xmax
    ymax = bb[3] / size[0]  # ymax

    return np.array([[ymin, xmin, ymax, xmax]])


def get_bbox(b4attribute):
    """

    :param b4attribute:
    :return: list
    """
    return [int(b4attribute.find_all('xmin')[0].text),
            int(b4attribute.find_all('ymin')[0].text),
            int(b4attribute.find_all('xmax')[0].text),
            int(b4attribute.find_all('ymax')[0].text)]

