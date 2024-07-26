import os
import numpy as np
import pandas as pd
import copy
import cv2
import math
import matplotlib.pyplot as plt

from ..datacubepredictors.segmentation import MASKRCNN_Detector
from ..utils.distances import euclidean_distance

from ..cropcv.detection_plots import plot_segmenimages, random_colors, draw_frame

from .metrics import ShapeMetricsFromMaskedLayer
from .utils import calculate_quantiles, from_quantiles_dict_to_df

from ..cropcv.image_functions import to_standard_rgb, transformto_cielab, transformto_hsv

class SeedsCounter(ShapeMetricsFromMaskedLayer, MASKRCNN_Detector):
    """
    A class for high-throughput seed phenotyping.

    Parameters
    ----------
    detector : torch.nn.Module
        The pre-trained detector model.
    detector_size : tuple of int, optional
        Size of the detector input.
    transform : callable, optional
        Transformation function to apply to the images.
    device : str, optional
        Device to run the model on ('cuda:0' or 'cpu').

    Attributes
    ----------
    _img : np.ndarray
        The input image.
    _colors : dict
        Dictionary of colors for seeds.
    _segmentation_thresholds : dict
        Thresholds for segmentation and prediction.
    """
    def __init__(self, detector,
                 detector_size = None,
                        transform=None,
                        device = None) -> None:
        
        self._img = None
        self._colors = None

        
        MASKRCNN_Detector.__init__(self,model = detector, transform = transform, device = device, input_size = detector_size)
        
    @property
    def available_color_spaces(self):
        """
        Available color spaces for transformation.

        Returns
        -------
        dict
            Dictionary mapping color space names to their transformation functions.
        """
        
        return {
            'cielab': [transformto_cielab,['l','a','b']],
            'srgb': [to_standard_rgb,['r','g','b']],
            'hsv': [transformto_hsv,['h','s','v']]
        }
    
    @property
    def seed_colors(self):
        """
        Colors assigned to each seed.

        Returns
        -------
        dict
            Dictionary with seed colors and labels.
        """
        
        if self._colors is None:
            colors = random_colors(self.msks.shape[0])
            colors = {i:{'color': np.array(c),
             'label': str(i)} for i,c in enumerate(colors)}
            self._colors =colors

        return self._colors
            
    
    def visualize_detected_seeds(self, label_factorsize:int = 50, **kwargs):
        """
        Visualize detected seeds.

        Parameters
        ----------
        label_factorsize : int, optional
            Size factor for labels.

        Returns
        -------
        np.ndarray
            Image with visualized seeds.

        Raises
        ------
        Exception
            If detection has not been run.
        """
        
        if self._img is not None:
            img_c = copy.deepcopy(self._img)
            for i in range(self.msks.shape[0]):
                    img_c = plot_segmenimages(img_c, self.msks[i], 
                                            boxes = None, 
                                            invert_rgb_order=False, only_image=True,
                                            mask_color=self.seed_colors[i]['color'])
            
            img_c = draw_frame(img_c, self.bbs, bbtype = 'xminyminxmaxymax', dictlabels=self.seed_colors, sizefactorred = label_factorsize,  **kwargs)
            return img_c
        else:
            raise Exception('Run detection first')
    
        
    def calculate_color_space_values(self, color_spacelist: list, include_srgb = True) -> dict:
        """
        Calculate color space values for a seed.

        Parameters
        ----------

        color_spacelist : list of str
            List of color spaces to calculate (e.g., 'cielab', 'hsv').

        Returns
        -------
        dict
            Dictionary with color space values.
        """

        srgb = to_standard_rgb(self._seedrgb)
        csimage = {}
        for colospacename in color_spacelist:
            fun = self.available_color_spaces[colospacename][0]
            csimage[colospacename] = fun(srgb)
        if include_srgb:
            srgb[srgb == 0] = np.nan
            csimage['srgb'] = srgb
        
        return csimage
    
    def calculate_color_metrics(self, quantiles: list = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        """
        Calculate color metrics for the seeds.

        Parameters
        ----------
        quantiles : list of float, optional
            List of quantiles to calculate (default is [0.25, 0.5, 0.75]).

        Returns
        -------
        pd.DataFrame
            DataFrame containing color metrics.
        """
        dfvals = []
        for k,v in self._seed_colorspace.items():
            channel_names = self.available_color_spaces[k][1]
            npcoloval = v.swapaxes(2,1).swapaxes(1,0) if v.shape[0]>10 else v
            quantile = calculate_quantiles(npcoloval, quantiles= quantiles)
            dfvals.append(from_quantiles_dict_to_df({channel_names[i]: v for i,v in enumerate(quantile)}, idvalue= 'seed_{}'.format(self._seedid)))
        
        dfc = pd.concat(dfvals, axis = 1).reset_index().drop(['id','index'], axis = 1)
        #dfc['seedid'] = 'seed_{}'.format(self._seedid)
        return dfc

    def _reset_seed_metrics(self) -> None:
        """
        Reset the seed metrics.
        """
        
        self._seedid = None
        self._seedrgb = None
        self._seedmask = None
        self._lengthpoints= None
        self._widthpoints = None
        self._seedcgcenter = None
        self._seediscenter = None
        self._perimetercoords = None
        self._seed_colorspace = None
        #self._colors = None    
    
    def single_seed_phenotyping(self, seedid:int, perpendicular_tolerance = 0.01, padding_percentage = 1, color_spacelist = ['hsv','cielab'], include_srgb = True):
        """Calculate all seed metrics
        in color the RGB information is transformed to standard rgb https://en.wikipedia.org/wiki/SRGB
        
        Parameters
        ----------
        seedid : int
            The ID of the seed.
        perpendicular_tolerance : float, optional
            Tolerance for perpendicular measurement (default is 0.001).
        padding_percentage : int, optional
            Padding as a percentage of the image size.
            
        """
        self._reset_seed_metrics()
        self._seedid = seedid

        self._seedrgb, self._seedmask = self._clip_rgb_and_mask(seedid, padding_percentage = padding_percentage, maskrgb = True)    
        self._lengthpoints, self._widthpoints = self.get_length_and_widths_coordinates(self._seedmask, perpendicular_tolerance=perpendicular_tolerance)
        _, self._seedcgcenter, self._seediscenter = self.distance_between_centers(mask_image = self._seedmask, 
                                                                                  length_width_coords = (self._lengthpoints, self._widthpoints),  
                                                                                  perpendicular_tolerance = perpendicular_tolerance)
        
        self._perimetercoords = self._get_perimeter_coords(self._seedmask)
        
        if color_spacelist is not None:
            self._seed_colorspace = self.calculate_color_space_values(color_spacelist=color_spacelist, include_srgb = include_srgb)
        
        
    
    def calculate_seed_morphometrics(self) -> dict:
        """
        Calculate various morphometric properties of a seed.


        Returns
        -------
        dict
            Dictionary containing length, width, perimeter, distance between centers,
            area, circularity, roundness, and length-to-width ratio of the seed.
        """
        #_, imgcc = self._clip_rgb_and_mask(seedid, **kwargs)
        
        ## get length and width    
        (ph0,ph1), (pw0, pw1) = self._lengthpoints, self._widthpoints
        length = euclidean_distance(ph0, ph1)
        width = euclidean_distance(pw0, pw1)
        ## calculate distance between intersection center and gravity center 
        # reference: https://doi.org/10.1104/pp.112.205120
        dbwcenters = euclidean_distance(self._seedcgcenter, self._seediscenter)
        
        ## calculate perimeter and area
        countour = self._countours(self._seedmask)[0]
        perimeter = cv2.arcLength(countour,True)
        #area = np.sum(self._seedmask>1)
        area = cv2.contourArea(countour)
        ## length width ration
        lwr = length/width
        ## circularity
        cs = (4*math.pi*area)/(perimeter*perimeter)
        ## roundness 
        # reference: https://doi.org/10.1016/j.anres.2017.12.002
        roundness = (4*math.pi*area)/(perimeter)
        
        return {'length':length,
            'width':width,
            'perimeter': perimeter,
            'distance_between_centers': dbwcenters,
            'area': area,
            'circularity':cs, 
            'roundness': roundness,
            'length_to_width_ratio': lwr}
    
    def _clip_rgb_and_mask(self, seedid: int, padding: int = None, padding_percentage: int = 1, maskrgb: bool = True) -> tuple:
        """
        Clip RGB image and mask around a seed.

        Parameters
        ----------
        seedid : int
            Seed ID.
        padding : int, optional
            Padding around the seed.
        padding_percentage : int, optional
            Padding as a percentage of the image size.
        maskrgb : bool, optional
            Whether to mask the RGB image with the seed mask.

        Returns
        -------
        tuple
            Clipped RGB image and mask.
        """
        if padding is None: 
            padding = int(np.array(self._img).shape[0]*padding_percentage/100)
        img_c = copy.deepcopy(self._img)
        rgbimageclipped = self._clip_image(img_c, self.bbs[seedid], padding=padding,padding_with_zeros=False)
        maskclipped = self._clip_image(self.msks[seedid], self.bbs[seedid], padding=padding,padding_with_zeros=False)
        if maskrgb:
            rgbimageclipped[maskclipped<1] = 0
        
        return rgbimageclipped, maskclipped
    
    def plot_individual_seed(self, ax = None) -> plt.Axes:
        """
        Plot individual seed.

        Parameters
        ----------
        ax : matplotlib.axes
            

        Returns
        -------
        plt.Axes
            Matplotlib axes with the plotted seed.
        """

        perimeter_coords = self._perimetercoords.T
        
        (ph0,ph1), (pw0, pw1) = self._lengthpoints, self._widthpoints
        (cgx, cgy), (isx, isy) = self._seedcgcenter, self._seediscenter 
        
        if ax is None:
            ax = plt.subplot()
        
        ax.imshow(self._seedrgb)
        ax.plot(perimeter_coords[0],perimeter_coords[1], c = self.seed_colors[self._seedid]['color'], linewidth = 4)
        ax.plot((ph0[0] , ph1[0]), (ph0[1],ph1[1]), 'o--', c = 'green', linewidth = 2)
        ax.plot((pw0[0] , pw1[0]), (pw0[1],pw1[1]), 'o--', c = 'purple', linewidth = 2)
        ax.plot((cgx,isx),(cgy , isy), ':', c = 'gray', linewidth = 3)
        ax.scatter(isx,isy, c = 'black')
        ax.scatter(cgx,cgy, c = 'red')
        
        return ax
    
    def get_all_seed_metrics(self, perpendicular_tolerance: float = 0.01, padding_percentage: float = 1, color_spacelist: list = ['hsv', 'cielab'], include_srgb: bool = True, quantiles: list = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        """
        Get all metrics for all detected seeds.

        Parameters
        ----------
        perpendicular_tolerance : float, optional
            Tolerance for perpendicular measurement (default is 0.001).
        padding_percentage : float, optional
            Percentage of padding to add (default is 1).
        color_spacelist : list of str, optional
            List of color spaces to include (default is ['hsv', 'cielab']).
        include_srgb : bool, optional
            Whether to include sRGB color space (default is True).
        quantiles : list of float, optional
            List of quantiles to calculate (default is [0.25, 0.5, 0.75]).

        Returns
        -------
        pd.DataFrame
            DataFrame containing all metrics for all seeds.
        """
        
        allseeds = []
        if not len(self.bbs)>0:
            raise ValueError('No seeds were detected')
            
        for i in range(len(self.msks)):
            self.single_seed_phenotyping(seedid=i, perpendicular_tolerance=perpendicular_tolerance, 
                                         padding_percentage=padding_percentage, color_spacelist= color_spacelist, include_srgb=include_srgb)
            
            dfs = pd.DataFrame(self.calculate_seed_morphometrics(), index=[0])
            if self._seed_colorspace is not None:
                colorvals = self.calculate_color_metrics(quantiles=quantiles)
                dfs = pd.concat([dfs,colorvals], axis = 1).reset_index()
                
            dfs['seedid'] = 'seed_{}'.format(self._seedid)
            allseeds.append(dfs)
        
        dfc = pd.concat(allseeds, axis = 0).reset_index().drop(['level_0','index'], axis = 1)
        return dfc
    
    def plot_all_seeds_metrics(self, ncols: int = 4, perpendicular_tolerance: float = 0.001, padding_percentage: float = 1, export_path: str = None, figsize: tuple = (15, 15)) -> plt.Figure:
        """
        Plot metrics for all detected seeds.

        Parameters
        ----------
        ncols : int, optional
            Number of columns in the plot grid (default is 4).
        perpendicular_tolerance : float, optional
            Tolerance for perpendicular measurement (default is 0.001).
        padding_percentage : float, optional
            Percentage of padding to add (default is 1).
        export_path : str, optional
            Path to export the plot image (default is None).
        figsize : tuple of int, optional
            Size of the figure (default is (15, 15)).

        Returns
        -------
        plt.Figure
            Matplotlib Figure object with the plotted metrics.
        """
        fig = plt.figure(figsize= figsize)
        
        nrows = len(self.msks)//ncols if len(self.msks) % ncols == 0 else (len(self.msks)//ncols )+1
        
        if not len(self.bbs)>0:
            raise ValueError('No seeds were detected')
        
        for i in range(len(self.msks)):
            self.single_seed_phenotyping(seedid=i, perpendicular_tolerance=perpendicular_tolerance, 
                                               padding_percentage=padding_percentage, color_spacelist= None)

            ax = fig.add_subplot(ncols,nrows, i+1)
            ax.set_axis_off()
            ax.set_title(f'{self._seedid}')
            
            self.plot_individual_seed(ax=ax)
            
        if export_path:
            fig.savefig(export_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            return fig
    
   
    def detect_seeds(self, image: np.ndarray, segmentation_threshold: float = 50, prediction_threshold: float = 0.75) -> None:
        """
        Detect seeds in an image.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        segmentation_threshold : float, optional
            Threshold for segmentation.
        prediction_threshold : float, optional
            Threshold for prediction.

        """
        self._reset_seed_metrics()
        self._colors = None
        
        self.detect_layers(image_data=image,img_path=None,threshold=prediction_threshold, 
                           segment_threshold = segmentation_threshold )
        

        self.seed_colors
        
        
        
        
