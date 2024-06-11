import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np

from numpy.core.fromnumeric import size
import colorsys
import random


def _apply_mask(image: np.ndarray, mask:np.ndarray, 
                color: List[float], alpha: float =0.5):
    """
    Apply color with alpha blending to the image based on the mask.

    Parameters:
    -----------
    image : np.ndarray
        Input image. order HWC
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



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def add_label(img: np.ndarray,
              label: str,
              xpos: int,
              ypos: int,
              font: int = None,
              fontcolor: tuple = None,
              linetype: int = 2,
              fontscale: float = None,
              thickness: int = 1) -> np.ndarray:
    """
    Adds a label to an image at the specified position.

    Parameters
    ----------
    img : np.ndarray
        The image to which the label will be added.
    label : str
        The text of the label to be added.
    xpos : int
        The x-coordinate of the label's position.
    ypos : int
        The y-coordinate of the label's position.
    font : int, optional
        The font type to be used for the label. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
    fontcolor : tuple, optional
        The color of the font in BGR format. Defaults to (0, 0, 0).
    linetype : int, optional
        The type of line for the font. Defaults to 2.
    fontscale : float, optional
        The scale factor for the font size. Defaults to 0.3.
    thickness : int, optional
        The thickness of the font. Defaults to 1.

    Returns
    -------
    np.ndarray
        The image with the label added.
    """
    
    fontscale = fontscale or 0.3
    fontcolor = fontcolor or (0,0,0)
    font = font or cv2.FONT_HERSHEY_SIMPLEX    

    img = cv2.putText(img,
            label, 
            (xpos, ypos), 
            font, 
            fontscale,
            fontcolor,
            thickness,
            linetype)

    return img
    

def add_frame_label(imgc: np.ndarray,
                    label: str,
                    coords: tuple,
                    color: tuple = None,
                    sizefactorred: int = 200,
                    heightframefactor: float = 0.2,
                    widthframefactor: float = 0.8,
                    frame: bool = True,
                    textthickness: int = 1) -> np.ndarray:
    """
    Adds a framed label to a specific area of an image.

    Parameters
    ----------
    imgc : np.ndarray
        The image to which the label will be added.
    label : str
        The text of the label to be added.
    coords : tuple
        Coordinates of the rectangular area in the format (x1, y1, x2, y2).
    color : tuple, optional
        The color of the frame and text in BGR format. Defaults to (255, 255, 255).
    sizefactorred : int, optional
        The size reduction factor for the text. Defaults to 200.
    heightframefactor : float, optional
        The height factor for the frame. Defaults to 0.2.
    widthframefactor : float, optional
        The width factor for the frame. Defaults to 0.8.
    frame : bool, optional
        Whether to add a frame around the label. Defaults to True.
    textthickness : int, optional
        The thickness of the text. Defaults to 1.

    Returns
    -------
    np.ndarray
        The image with the framed label added.
    """
    
    color = (255,255,255) if color is None else color
    x1,y1,x2,y2 = coords
    
    widhtx = abs(int(x1) - int(x2))
    heighty = abs(int(y1) - int(y2))
        
    xtxt = x1 if x1 < x2 else x2
    ytxt = y1 if y1 < y2 else y2
    
    if frame:
        imgc = cv2.rectangle(imgc, (xtxt,ytxt), (xtxt + int(widhtx*widthframefactor), 
                                                     ytxt - int(heighty*heightframefactor)), color, -1)
        color = (255,255,255)
        
    imgc = cv2.putText(img=imgc, text=label,org=( xtxt + int(widhtx/15),
                                                          ytxt - int(heighty/20)), 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                                fontScale=1*((heighty)/sizefactorred), color=color, 
                                thickness=textthickness)
    
    return imgc
            

def draw_frame(img, bbbox: List, 
               dictlabels: Dict = None, 
               default_color = None, bbtype = None,
               heightframefactor = 0.2, 
               textthickness = 1,
               widthframefactor = 0.8,
               sizefactorred = 200, bb_thickness = 4):
    """
    Draw bounding boxes and labels on an image.

    Args:
        img (numpy.ndarray): Input image.
        bbbox (list): List of bounding boxes in the format [x1, y1, x2, y2] or [x1, x2, y1, y2].
        dictlabels (dict, optional): Dictionary containing labels and colors for each bounding box.
        default_color (list, optional): Default color for bounding boxes. The colors must be a list with RGB values with a length equal to the number of bounding boxes. Default is None.
        bbtype (str, optional): Bounding box type. Either 'xminyminxmaxymax' or 'xxyy'. Default is None.
        sizefactorred (int, optional): Size factor for the label frame. Default is 200.
        bb_thickness (int, optional): Thickness of the bounding box lines. Default is 4.

    Returns:
        numpy.ndarray: Image with bounding boxes and labels drawn.
    """
    
    imgc = img.copy()
    
    #get colors
    if default_color is None:
        default_color = [(1,1,1)]*len(bbbox)
    if not isinstance(default_color[0], list):
        default_color = [default_color]
        #print(default_color)
    for i in range(len(bbbox)):
        
        if bbtype == 'xminyminxmaxymax':
            x1,y1,x2,y2 = bbbox[i]
            
        else:
            x1,x2,y1,y2 = bbbox[i]

        start_point = (int(x1), int(y1))
        end_point = (int(x2),int(y2))
        if dictlabels is not None:
            color = dictlabels[i]['color']
            label = dictlabels[i]['label']
        else:
            label = str(i)
            color = default_color[i]
            
        if np.average(color) <= 1:
            color = [int(z*255) for z in color]
        
        imgc = cv2.rectangle(imgc, start_point, end_point, color, bb_thickness)
        if label != '':
            imgc = add_frame_label(imgc,
                    label,
                    [int(x1),int(y1),int(x2),int(y2)],color,sizefactorred,
                    heightframefactor = heightframefactor,
                    textthickness = textthickness,
                    widthframefactor = widthframefactor)
            
            
    return imgc   


def plot_segmenimages(image: np.ndarray, mask_image: np.ndarray, 
                      boxes: list = None, figsize: tuple = (10, 8),
                        bb_type: str = None, only_image: bool = False, 
                        invert_rgb_order: bool = True,
                        fontsize: int = 18, turn_off_axis: bool = True,
                        mask_color = None, alpha = 0.5, **kwargs)  -> plt.Figure:
    
    """
    Plot segmented images with optional bounding boxes.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    mask_image : np.ndarray
        Segmentation mask image.
    boxes : list, optional
        List of bounding boxes in format [x1, y1, x2, y2]. Defaults to None.
    figsize : tuple, optional
        Figure size. Defaults to (10, 8).
    bb_type : str, optional
        Type of bounding boxes. Either 'xminyminxmaxymax' or 'xyxy'. Defaults to None.
    only_image : bool, optional
        If True, only the segmented image is returned without additional plots. Defaults to False.
    invert_rgb_order : bool, optional
        If True, invert the RGB order of the input image. Defaults to True.
    fontsize : int, optional
        Font size for titles. Defaults to 18.
    turn_off_axis : bool, optional
        If True, turn off axes for subplots. Defaults to True.

    Returns:
    --------
    plt.Figure
        Matplotlib figure object.
    """
    
    if mask_color is None:
        datato = image.copy()
        heatmap = cv2.applyColorMap(np.array(mask_image).astype(np.uint8), 
                                    cv2.COLORMAP_PLASMA)
        
        output = cv2.addWeighted(datato, 0.5, heatmap, 1 - 0.75, 0)
    else:
        datato = image.copy()
        mask_imageones = np.array(mask_image).astype(np.uint8)
        mask_imageones[mask_imageones>0] = 1
        
        output = _apply_mask(datato, mask_imageones, mask_color, alpha=alpha)

    if boxes is not None:

        output = draw_frame(output, boxes, bbtype = bb_type, default_color = mask_color, **kwargs)
    
    if only_image:
        fig = output
    else:
        
        # plot the images in the batch, along with predicted and true labels
        fig, ax = plt.subplots(nrows = 1, ncols = 3,figsize=figsize)
        #ax = fig.add_subplot(1, fakeimg.shape[0], idx+1, xticks=[], yticks=[])
        #fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (14,5))
        #.swapaxes(0,1).swapaxes(1,2).astype(np.uint8)
        if invert_rgb_order:
            order = [2,1,0]
        else:
            order = [0,1,2]
            
        ax[0].imshow(datato[:,:,order],vmin=0,vmax=1)
        ax[0].set_title('Real',fontsize = fontsize)
        
        #ax[0].set_axisoff()
        ax[1].imshow(mask_image,vmin=0,vmax=1)
        ax[1].set_title('Segmentation',fontsize = fontsize)

        ax[2].set_title('Overlap',fontsize = fontsize)
        ax[2].imshow(output[:,:,order])
        if turn_off_axis:
            for axis in ax:
                axis.axis('off')
        
    return fig


def plot_single_image_odlabel(npimages: np.dstack, bbcoords = None,figsize = (12,10), linewidth = 2, edgecolor = 'r')->None:
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    ax.imshow(npimages)
    for i in range(len(bbcoords)):
        if bbcoords is not None and len(bbcoords[i])==4:
            
            
            x1,x2,y1,y2 = bbcoords[i]
            centerx = x1+np.abs((x1-x2)/2) 
            centery = y1+np.abs((y1-y2)/2) 
            
            rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
            ax.scatter(x=centerx, y=centery, c='r', linewidth=2)
            ax.add_patch(rect)
            
            
def plot_single_image(npimages: np.ndarray, idimage: int = 0, figsize: tuple = (12, 10)) -> None:
    """
    Plot a single image from a stack of NumPy arrays.

    Parameters
    ----------
    npimages : numpy.ndarray
        A 3D NumPy array containing the images.
    idimage : int, optional
        The index of the image to plot. Defaults to 0.
    figsize : tuple, optional
        The size of the figure to create. Defaults to (12, 10).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.imshow(npimages[idimage])
    plt.show()
    

    