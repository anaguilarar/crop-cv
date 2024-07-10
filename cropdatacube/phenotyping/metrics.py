import math
import copy
import xarray


import cv2
import numpy as np


from ..cropcv.image_functions import contours_from_image, apply_color_alpha, clip_image_usingbb

from ..cropcv.detection_plots import random_colors, add_frame_label
from ..cropcv.mask_layer_fun import get_boundingboxfromseg
from ..cropcv.mask_layer_fun import getmidleheightcoordinates, getmidlewidthcoordinates

from ..spatialdatacube.datacube_processors import DataCubeProcessing
from ..utils.distances import euclidean_distance, calculate_distance_matrix

from .utils import calculate_quantiles, perpendicular_points_to_line, line_intersection
from typing import Tuple, Optional, Dict, List


class ShapeMetricsFromMaskedLayer:
    """
    A class for computing metrics from an image layer, including adding metric lines,
    labels, and calculating height, width, and area from contours.
    """
    def calculate_indlayer_metrics(self, mask_pos, padding = 20, hull = False):
        """
        Calculate individual mask layer metrics for a given position.

        Args:
            mask_pos (int): segmentation mask position to calculate metrics for.
            padding (int, optional): Padding around the mask. Defaults to 20.
            hull (bool, optional): Whether to use convex hull for contours. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing calculated metrics.
        """
        maskimage = self._clip_image(self.msks[mask_pos], self.bbs[mask_pos], padding = padding)
        wrapped_box = self._find_contours(maskimage, hull = hull)
        pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(wrapped_box)
        d1 = euclidean_distance(pheightu, pheigthb)
        d2 = euclidean_distance(pwidthu, pwidthb)
        #distper = np.unique([euclidean_distance(wrapped_box[i],wrapped_box[i+1]) for i in range(len(wrapped_box)-1) ])
        # Determine larger and shorter dimensions
        larger = d1 if d1>d2 else d2
        shorter = d1 if d1<d2 else d2
        msksones = maskimage.copy()
        msksones[msksones>0] = 1
        
        area = np.sum(msksones*1.)
        return {
            'mask_id':[mask_pos],
            'height': [larger],
            'width': [shorter],
            'area': [area]}
        
        
    def image_layers_summary(self):
        import pandas as pd
        summarylist = []
        for i in range(len(self.bbs)):
            try: 
                summarylist.append(
                pd.DataFrame(self.calculate_indlayer_metrics(i)))
            except:
                pass
        if len(summarylist) > 0:
            summarylist = pd.concat(summarylist)
        else:
            summarylist = None
            
        return summarylist
    
    
    def _add_metriclines_to_single_detection( self, 
        rgb_image: np.ndarray, 
        mask_image: np.ndarray,
        add_lines: bool = True, 
        add_label: bool = True,
        padding: int = 30,
        mask: bool = False,
        size_factor_red: int = 250,
        height_frame_factor: float = .15,
        width_frame_factor: float = .3, 
        hull: bool = False,
        col: Optional[List[int]] = None,
        text_thickness: int = 1,
        line_thickness: int = 1
        ) -> np.ndarray:
        
        """
        Add lines and labels to a single detection in the given RGB and mask images.

        Parameters
        ----------
        rgb_image : np.ndarray
            The RGB image as a numpy array.
        mask_image : np.ndarray
            The mask image as a numpy array.
        add_lines : bool, optional
            A boolean indicating whether to add metric lines.
        add_label : bool, optional
            A boolean indicating whether to add labels.
        padding : int, optional
            The padding to use.
        mask : bool, optional
            A boolean indicating whether to apply the mask.
        size_factor_red : int, optional
            The size factor for reduction.
        height_frame_factor : float, optional
            The height frame factor.
        width_frame_factor : float, optional
            The width frame factor.
        hull : bool, optional
            A boolean indicating whether to use the convex hull.
        col : Optional[List[int]], optional
            The color for the metrics as a list of RGB values. Defaults to white if None.
        text_thickness : int, optional
            The thickness of the label text.
        line_thickness : int, optional
            The thickness of the metric lines.

        Returns
        -------
        np.ndarray
            A numpy array of the image with metric lines and labels added.
        """
        
        col = [1,1,1] if col is None else col

        imageres = copy.deepcopy(rgb_image)
        msksones = copy.deepcopy(mask_image)

        msksones[msksones>0] = 1
        msksones = msksones.astype(np.uint8)
        
        newimg = cv2.bitwise_and(imageres,
                                 imageres, mask = msksones) if mask else imageres
        
        img = apply_color_alpha(newimg, msksones, col, alpha=0.2).astype(np.uint8)
        
        linecolor = list((np.array(col)*255).astype(np.uint8))
        imagewithborders = cv2.drawContours(img,[self._find_contours(mask_image, hull = hull)],
                             0,[int(i) for i in linecolor],
                            line_thickness)
        if add_lines:
            pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(
                self._find_contours(mask_image, hull = hull))
            imagewithborders = cv2.line(imagewithborders, pheightu, pheigthb, (0,0,0), line_thickness)
            imagewithborders = cv2.line(imagewithborders, pwidthu, pwidthb, (0,0,0), line_thickness)

        if add_label:
            x1,y1,x2,y2 = get_boundingboxfromseg(mask_image)

            imagewithborders = add_frame_label(imagewithborders, str(0), [int(x1),int(y1),int(x2),int(y2)],[
                int(i*255) for i in col],
                    sizefactorred = size_factor_red,
                    heightframefactor = height_frame_factor,
                    widthframefactor = width_frame_factor,
                    textthickness = text_thickness)

        return imagewithborders
    
    def calculate_hwa_from_contours(self, 
            mask_image: Optional[np.ndarray] = None, 
            pixel_factor: Optional[float] = None,
            hull: bool = False
        ) -> Dict[str, List[float]]:
        """
        Calculates height, width, and area from the contours of a mask image.

        Parameters
        ----------
        mask_image : Optional[np.ndarray], optional
            The mask image as a numpy array. If None, an empty dictionary is returned.
        pixel_factor : Optional[float], optional
            The factor by which to scale the dimensions to obtain real-world units. If None, dimensions are returned in pixels.
        hull : bool, optional
            A boolean indicating whether to apply the convex hull.
        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing 'height', 'width', and 'area' with the calculated values.
        """
        if mask_image is None:
            return {}
        
        wrapped_box = self._find_contours(mask_image, hull = hull)
        pheightu, pheigthb, pwidthu, pwidthb = self._get_heights_and_widths(wrapped_box)
        
        d1 = euclidean_distance(pheightu, pheigthb)
        d2 = euclidean_distance(pwidthu, pwidthb)

        larger = d1 if d1>d2 else d2
        shorter = d1 if d1<d2 else d2

        msksones = mask_image.copy()
        msksones[msksones>0] = 1
        area = np.sum(msksones*1.)
        if pixel_factor:
            return { 'height': [larger * pixel_factor], 'width': [shorter* pixel_factor],'area': [area * (pixel_factor**2)]}
        else:
            return { 'height': [larger], 'width': [shorter],'area': [area]}
    
    def get_length_and_widths_coordinates(self,mask_image: np.ndarray, perpendicular_tolerance = 0.01):
        
        ## get perimeter coordinates
        countour_coords = self._get_perimeter_coords(mask_image)
        ## calculate length base on countour distance
        pl0, pl1 = self._get_max_length(countour_coords)
        linecoord = np.squeeze((pl0, pl1)).flatten()   
        ## calculate width 
        countour_coords = self._get_perimeter_coords(mask_image)
        pw0, pw1 = self._get_width(countour_coords, linecoord, diff_factor= perpendicular_tolerance)
        
        return [pl0, pl1], [pw0, pw1]
        
    
    @staticmethod
    def distance_between_centers(mask_image = None, countour_coords = None, length_width_coords = None, perpendicular_tolerance = 0.01):
    
        if countour_coords is None:
            countour_coords = ShapeMetricsFromMaskedLayer()._get_perimeter_coords(mask_image)
        
        if length_width_coords is not None:
            (ph0,ph1),(pw0,pw1) = length_width_coords
        else:
            (ph0,ph1),(pw0,pw1) = ShapeMetricsFromMaskedLayer().get_length_and_widths_coordinates(mask_image, 
                                                                                                  perpendicular_tolerance = perpendicular_tolerance)
            
        cgx, cgy = np.mean(countour_coords, axis = 0)

        isx, isy = line_intersection((ph0,ph1),(pw0,pw1))
        
        return euclidean_distance((cgx, cgy), (isx, isy)), (cgx, cgy), (isx, isy)

        
    @staticmethod
    def _get_perimeter_coords(mask_image):
        contours = ShapeMetricsFromMaskedLayer()._countours(mask_image)
        return np.array([np.squeeze(contours[0][i]) for i in range(len(contours[0]))])
        
    
    @staticmethod
    def _get_max_length(countour_coords):
        ## calculate pair distances
        countour_distances = calculate_distance_matrix(countour_coords, countour_coords)
        ## the maxixum distance will be the length
        maxlpos = np.where(countour_distances == np.max(countour_distances))
        maxlpos = (maxlpos[0][0],maxlpos[1][0]) if len(maxlpos[0])>1 else maxlpos
        return np.squeeze(countour_coords[maxlpos[0]]),np.squeeze(countour_coords[maxlpos[1]])
    
    @staticmethod
    def _get_width(countor_coords, linecoords, diff_factor = 0.01):
        #diff_factor = 0.001
        perppoints = []
        while len(perppoints)==0 and diff_factor < 1:
            perppoints = perpendicular_points_to_line(linecoords, countor_coords, diff_factor = diff_factor)
            diff_factor= diff_factor*10
        
        perpdistances = [euclidean_distance(p1,p2) for p1, p2 in perppoints]
        
        maxlpos = np.where(perpdistances == np.max(perpdistances))[0]

        maxlpos = maxlpos[0] if len(maxlpos)>0 else maxlpos
        
        return np.squeeze(perppoints[maxlpos])
            

    
    
    @staticmethod
    def _get_heights_and_widths(mask_contour_points):
        """
        Gets the heights and widths from mask contours.

        Parameters
        ----------
        mask_contour_points : np.ndarray
            The contour vertices of the mask as a numpy array.

        Returns
        -------
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
            A tuple containing coordinates for upper height, bottom height, upper width, and bottom width.
        """
        # Implementation note: Requires correction for euclidean_distance and get_middle_*_coordinates function calls.
        
        p1,p2,p3,p4=mask_contour_points
        alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

        pheightu=getmidleheightcoordinates(p2,p3,alpharad)
        pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
        pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
        pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

        return pheightu, pheigthb, pwidthu, pwidthb
    
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
    
    @staticmethod 
    def _countours(image):
        maskimage = image.copy()
        contours = contours_from_image(maskimage)
        
        return contours
    
    @staticmethod 
    def _find_contours(image, hull = False):
        """
        Finds contours in an image, optionally applying a convex hull.

        Parameters
        ----------
        image : np.ndarray
            The image as a numpy array.
        hull : bool, optional
            A boolean indicating whether to apply the convex hull.

        Returns
        -------
        np.ndarray
            The contours as a numpy array.
        """
        
        contours = ShapeMetricsFromMaskedLayer()._countours(image)
        
        if hull:
            firstcontour = cv2.convexHull(contours[0])
        else:
            firstcontour = contours[0]
        
        rect = cv2.minAreaRect(firstcontour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    


class DataCubePhenomics(DataCubeProcessing):
    """
    A class for summarizing data cube, specifically tailored for
    generating quantile summaries of 2D and 3D image channels.

    Attributes
    ----------
    xrdata : xarray.Dataset
        The xarray dataset containing the phenomic data.
    _channel_names : List[str]
        A list of channel names available in the dataset.
    _array_order : str
        The order of dimensions in the dataset, e.g., 'CHW' for channels, height, width.
    _ndims : int
        The number of dimensions in the dataset.
    _navalue : float
        The value used in the dataset to represent N/A or missing data.
    """
    def __init__(self, xrdata: xarray.Dataset = None, metrics: dict = None, 
                 array_order:str = 'CHW', navalue: float = 0) -> None:
        """
        Initializes the PhenomicsFromDataCube instance with the provided xarray dataset and configuration.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset containing the phenomic data.
        metrics : Optional[Dict], optional
            A dictionary of metrics for summarization, by default None.
        array_order : str, optional
            The order of dimensions in the dataset, e.g., 'CHW' for channels, height, width, by default 'CHW'.
        navalue : float, optional
            The value used in the dataset to represent N/A or missing data, by default np.nan.
        """
        self.xrdata = xrdata
        self._array_order = array_order
        self._navalue = navalue
        self._update_params()
    
    def __call__(self, xrdata: xarray.Dataset, 
                 channels: Optional[List[str]] = None,
                 vegetation_indices: Optional[List[str]] = None,
                 color_spaces: Optional[List[str]] = None,
                 quantiles: Optional[List[float]] = None,
                 rgb_channels: Optional[List[str]] = ['red','green','blue']) -> Dict[str, Dict[float, float]]:
        """
        Makes the instance callable, allowing it to directly operate on an xarray.Dataset
        to summarize its data into quantiles for specified channels.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset to be analyzed and summarized.
        channel_names : Optional[List[str]], optional
            A list of channel names to be summarized. If None, all channels will be summarized, by default None.
        quantiles : Optional[List[float]], optional
            A list of quantiles to calculate for each specified channel. If None, defaults to median (0.5), by default None.
        rgbchannels : List[str], optional
            List of channel names representing RGB. Defaults to ['red', 'green', 'blue'].
        Returns
        -------
        Dict[str, List[Dict[float, float]]]
            A dictionary where keys are channel names and values are lists of dictionaries, each containing calculated quantile values.
        """
        
        self.xrdata = xrdata
        channels = [] if channels is None else channels
        color_spaces = [] if color_spaces is None else color_spaces
        vegetation_indices = [] if vegetation_indices is None else vegetation_indices
        
        ## check vi list
        vegetation_indices = vegetation_indices if isinstance(vegetation_indices, list) else [vegetation_indices]
        vitocalculate = [channelname for channelname in vegetation_indices if channelname in self._available_vi]
        if len(vitocalculate)>0:
            self.calculate_vegetation_indices(vi_list=vitocalculate, update_data=True)
            self._update_params()
            
        ## check colors
        color_spaces = color_spaces if isinstance(color_spaces, list) else [color_spaces]
        coloravail = list(self._available_color_spaces.keys())
        colortocalculate = [channelname for channelname in color_spaces if channelname in coloravail]
        color_channels = []
        if len(colortocalculate)>0:
            for i in colortocalculate:
                self.calculate_color_space(color_space = i, rgbchannels =rgb_channels , update_data=True)
                color_channels += self._available_color_spaces[i]
                
            self._update_params()
            
        channel_names = channels + vitocalculate + color_channels
        return self.summarise_into_quantiles(channel_names=channel_names, quantiles=quantiles)


    @property
    def _depth_dimname(self):
        """Identifies the name of the depth dimension in the dataset, excluding common spatial dimensions."""
        dimsnames = self.xrdata.dims.keys()
        if len(dimsnames) == 2:
            depthname = None
        else:
            depthname = [i for i in dimsnames if i not in ['x','y','longitude','latitude']][0]
        
        return depthname
    
        
    def summarise_into_quantiles(self,channel_names: Optional[List[str]] = None, quantiles: List[float] = None):
        """
        Summarizes the data of specified channels into quantiles for 2D and 3D images within the dataset.

        Parameters
        ----------
        channel_names : Optional[List[str]], optional
            A list of channel names to summarize. If None, all channels are used, by default None.
        quantiles : Optional[List[float]], optional
            A list of quantiles to calculate for each channel. If None, defaults to the median (0.5), by default None.

        Returns
        -------
        Dict[str, List[Dict[float, float]]]
            A dictionary where keys are channel names and values are lists of dictionaries, each containing calculated quantile values.
        """
        if not channel_names:
            channel_names = self._channel_names
        
        channel_data = {}
        for channel_name in channel_names:
            channel_name = self._channel_names[0] if not channel_name else channel_name
            
            if self._ndims == 2:
                channel_data[channel_name] = self._2d_array_summary(
                    self.xrdata, channel_name=channel_name, quantiles= quantiles)
            if self._ndims == 3:
                channel_data[channel_name] = self._3d_array_summary(
                    self.xrdata, channel_name=channel_name, quantiles= quantiles)
        
        return channel_data
    
    def _update_params(self):
        self._channel_names = None if not self.xrdata else list(self.xrdata.keys())
        self._ndims = None if not self.xrdata else len(list(self.xrdata.sizes.keys()))
        
    
    def _3d_array_summary(self, channel_name: str = None, quantiles: List[float] = None):
        """
        Summarizes 3D images into quantiles for a specific channel.

        Parameters
        ----------
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with quantile values for each depth slice of the 3D image.

        Raises
        ------
        ValueError
            If the dataset's array order is not 'DCHW', indicating depth, channel, height, width.
        """
        if self._array_order != "DCHW":
            raise ValueError('Currently implemented only for "DCHW" array order.')
        
        datasummary = []
        for i in self.xrdata.dims[self._depth_dimname]:
            xrdata2d = self.xrdata.isel({self._depth_dimname: i})
            datasummary.append(self._2d_array_summary(xrdata2d, channel_name))
        
        return datasummary
        
    
    def _2d_array_summary(self, xrdata: xarray.Dataset, channel_name: None, quantiles: List[float] = None):
        """
        Summarizes 2D images into quantiles for a specific channel.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset to summarize.
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with calculated quantile values.

        Raises
        ------
        AssertionError
            If the specified channel name is not in the datacube's channel names.
        """
        if not quantiles:
            quantiles = [0.5]

        assert channel_name in self._channel_names, "Channel name must be in the datacube."
        
        channel_image = xrdata[channel_name].values
        if not self._navalue:
            channel_image[channel_image == self._navalue] = np.nan
            
        datasummary = calculate_quantiles(channel_image, quantiles=quantiles)
        
        return datasummary
        

    