from .utils import xyxy_to_xywh, from_yolo_toxy
from .base_processors import CVDetector_base
from ..cropcv.detection_plots import draw_frame
from ..cropcv.image_functions import read_image_as_numpy_array, resize_npimage
from ..spatialdatacube.orthomosaic import OrthomosaicProcessor
from ..spatialdatacube.gis_functions import from_polygon_2bbox, from_bbxarray_2polygon, merge_spatial_features
from ..utils.general import find_postinlist


import geopandas as gpd
import numpy as np
import os
import pandas as pd
import xarray as xr

from tqdm import tqdm
from typing import Tuple, List, Dict



class Orthomosaic_ODetector(OrthomosaicProcessor):
    """
    A class for detecting objects of interest in UAV images using a given detector model.

    Parameters
    ----------
    detector : object
        Object detection model used for detecting objects.
    orthomosaic_path : str
        Path to the orthomosaic image.
    bands : List[str], optional
        List of bands to use from the orthomosaic image. Defaults to None.
    multiband_image : bool, optional
        Indicates whether the orthomosaic image is multiband. Defaults to False.
    bounds : Dict, optional
        Dictionary containing the bounding coordinates of the orthomosaic image. Defaults to None.
    device : str, optional
        Device to run the detection model on (e.g., 'cpu', 'cuda'). Defaults to None.
    """
    
    def __init__(self, detector, orthomosaic_path: str, bands: List[str] = None, 
                 multiband_image: bool = False, bounds: Dict = None,
                 device:str = None):
        
        super().__init__(orthomosaic_path, bands, multiband_image, bounds)
        self.device = device
        self.detector = detector
    
    def detect_oi_in_uavimage(self, tilesize: int = 512, overlap: List = None, 
                              aoi_limit: float = 0.15, threshold_prediction: float = 0.1):
        """
        Detect objects of interest in UAV images.

        Args:
            tilesize (int, optional): Size of the tiles for image splitting. Defaults to 512.
            overlap (List, optional): List of overlap values for tile splitting. Defaults to None.
            aoi_limit (float, optional): Minimum area of interest limit. Defaults to 0.15.
            threshold_prediction (float, optional): Minimum pprediction accuracy for the prediction. Defaults to 0.1.

        Returns:
            tuple: Detected boundary boxes and polygons.
        """
        
        self._tilesize = tilesize
        overlap = [0] if overlap is None else overlap
        allpols_pred = []
        for spl in overlap:
            self.split_into_tiles(width = tilesize, height = tilesize, overlap = spl)
            ## only tiles with data
            onlythesetiles = []
            for i in tqdm(range(len(self._tiles_pols))):
                if np.sum(self.tiles_data(i)[self._bands[0]].values) != 0.0:
                    bbasgeodata = self.detect(self.tiles_data(i), threshold = threshold_prediction)
                
                    if bbasgeodata is not None:
                        bbasgeodata['tile']= [i for j in range(bbasgeodata.shape[0])]
                        allpols_pred.append(bbasgeodata)
                    
        if len(allpols_pred) > 0:
            allpols_pred_gpd = pd.concat(allpols_pred)
            allpols_pred_gpd['id'] = [i for i in range(allpols_pred_gpd.shape[0])]
            print("{} polygons were detected".format(allpols_pred_gpd.shape[0]))

            total_objects = merge_spatial_features(allpols_pred_gpd, mininterectedgeom = aoi_limit)
            
            total_objects = pd.concat(total_objects) 
            print("{} bounding boxes were detected".format(total_objects.shape[0]))
        
            return total_objects, allpols_pred
    
    def detect(self, xrimage, threshold = 0.1):
        """
        Detect objects in an image.

        Args:
            xrimage (xr.Dataset or xr.DataArray): Image data in xarray format.
            threshold (float, optional): Detection threshold. Defaults to 0.1.

        Returns:
            pd.DataFrame: Detected objects with their attributes.
        """
        
        if isinstance(xrimage, xr.Dataset):
            tiledata = xrimage.to_array().values.astype(np.uint8)
        else:
            tiledata = xrimage.values.astype(np.uint8)
        # TODO: IMAGE ORDER CHANGE FOR OTHER DETECTORS
        tiledata = tiledata.swapaxes(0,1).swapaxes(1,2)[:,:,[2,1,0]]
        origsize = tiledata.shape[:2]
        detections = self.detector(image=tiledata, threshold = threshold)
        
        xyxylist = [[int(j * origsize[0]) for j in i] for i in detections[0]]
        
        crs_system = None if xrimage.attrs['crs'] is None else xrimage.attrs['crs']
        polsshp_list = []
        if len(xyxylist):
            for i in range(len(xyxylist)):
                bb_polygon = from_bbxarray_2polygon(xyxylist[i], xrimage)

                pred_score = np.round(detections[1][i] * 100, 3)

                gdr = gpd.GeoDataFrame({'pred': [i],
                                        'score': [pred_score],
                                        'geometry': bb_polygon},
                                    crs=crs_system)

                polsshp_list.append(gdr)
            return pd.concat(polsshp_list, ignore_index=True)
        


class YOLOV8_cropdetector(CVDetector_base):
    """
    A detector class using YOLOv8 for object detection, allowing customization of the model path, input size, and bounding box type.

    Parameters
    ----------
    model : Any, optional
        Preloaded YOLO model, by default None.
    path : str, optional
        Path to the YOLO model file, by default None.
    input_size : Tuple[int, int], optional
        Size to which input images should be resized, by default (640, 640).
    bbstype : str, optional
        Type of bounding boxes ('xywh' or 'xyxy'), by default "xywh".

    Raises
    ------
    ValueError
        If neither model directory path nor model is provided.
    """
    
    def __init__(self, 
                 model = None,
                 path: str = None,
                 input_size: Tuple[int, int] = (640, 640),
                 bbstype:str = "xywh") -> None:
        
        from ultralytics import YOLO
        
        self.inputsize = input_size
        if path is not None:
            assert os.path.exists(path) ## path to model does not exists
            self.model = YOLO(path)
        elif model is not None:
            self.model = model
        else:
            raise ValueError(" please provide either the directory path or the model")
        
        self.bbstype = bbstype

    def __call__(self,  **kwargs):
        """
        Enables the detector object to be called like a function, forwarding call to `detect_objects`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Detected bounding boxes and scores.
        """
        
        return self.detect_objects( **kwargs)
    
    def detect_objects(self,
        img_path: str = None,
        image: np.ndarray = None,
        threshold: float = 0.75) :
        
        """
        Detect objects in an image provided either as a path or as an ndarray.

        Parameters
        ----------
        img_path : Optional[str], optional
            Path to the image file, by default None.
        image : Optional[np.ndarray], optional
            Image as a numpy array, by default None.
        threshold : float, optional
            Confidence score threshold for filtering detections, by default 0.75.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Detected bounding boxes and scores filtered by the threshold.

        Raises
        ------
        ValueError
            If neither an np.array image nor a directory path is provided.
        """
        
        
        if img_path is not None:
            image = self.read_image(imgpath=img_path)
            
        elif image is not None:
            image = image #self.resize_image(image)
        else:
            raise ValueError("an np array image or a directory path must be provided")
        
        results = self.model(image, verbose=False)[0]
        if self.bbstype == "xywh":
            self.bbs = results.boxes.xywhn.detach().cpu().numpy()
        if self.bbstype == "xyxy":
            self.bbs = results.boxes.xyxyn.detach().cpu().numpy()
        
        self._scores = results.boxes.conf.detach().cpu().numpy()

        self._filter_byscore(threshold)
        
        return self.bbs, self._scores
        
    def _filter_byscore(self, 
                        threshold: float):
        
        """
        Filters the bounding boxes based on the confidence score threshold.

        Parameters
        ----------
        threshold : float
            Confidence score threshold.
        """
        
        onlythesepos = np.where(
            self._scores>threshold)
        
        self.bbs = self.bbs[onlythesepos]
        self._scores[onlythesepos]
        
    def resize_image(self, image):
        """
        Reads an image from the file and resizes it according to the detector's input size.

        Parameters
        ----------
        img_path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            The resized image as a numpy array.

        Raises
        ------
        AssertionError
            If the provided image path does not exist.
        """
        if self.inputsize is not None:
            image = resize_npimage(image, size= self.inputsize)
            self.keep_size = True
        else:
            self.keep_size = False
            
        return image
    
    def read_image(self, imgpath:str):
        """
        Read image from file.

        Args:
            img_path (str): Path to the image file.
        """
        assert os.path.exists(imgpath) ## path does not exist
        img = read_image_as_numpy_array(imgpath)
        self._origsize = img.shape[:2]
        if self.inputsize is not None:
            img = resize_npimage(img, size= self.inputsize)
            self.keep_size = True
        else:
            self.keep_size = False
            
        return img

class UAVBB_fromvector(OrthomosaicProcessor):
    """Class to create labels ferom spatial vector files. Currently only process one polygon at the time

    Args:
        OrthomosaicProcessor (class): Class created for processing uav imagery and keep them as xarray

    Returns:
        txt: yolo file
    """
    @property
    def xycoords(self):
        x1, y1, x2, y2 = from_polygon_2bbox(self.geom_polygon)
        return x1, y1, x2, y2
    
    @property
    def geom_polygon(self):
        assert type(self.boundary) is gpd.GeoDataFrame
        geompolygon = self.boundary.geometry.values[0]
        return geompolygon
    
    @property
    def image_coords(self):
        
        xcoords = self.drone_data.coords['x'].values.copy()
        ycoords = self.drone_data.coords['y'].values.copy()
        
        l = find_postinlist(xcoords, self.xycoords[0])
        b = find_postinlist(ycoords, self.xycoords[1])
        r = find_postinlist(xcoords, self.xycoords[2])
        t = find_postinlist(ycoords, self.xycoords[3])
        
        return [l,b,r,t]
    
    @property
    def xyhw_coords(self):
        return xyxy_to_xywh(self.image_coords)
    
    def yolo_style(self, labelid = None):
        imc= self.drone_data.copy().to_array().values.swapaxes(0,1).swapaxes(1,2)
        x,y,h,w = self.xyhw_coords

        labelid = 0 if labelid is None else labelid
        
        heigth = imc.shape[1]
        width = imc.shape[0]

        return [0, x/width,y/heigth,h/heigth,w/width]
            
    def bb_plot(self, labelid = None):
        """Draw a plot with the image

        Args:
            labelid (str): category's name. Defaults to None.

        Returns:
            numpy.array: image
        """
        labelid = 0 if labelid is None else labelid
        imc= self.drone_data.copy().to_array().values.swapaxes(0,1).swapaxes(1,2)
        
        imgdr = draw_frame(imc.copy(), 
                    [from_yolo_toxy(self.yolo_style(labelid), imc.shape[:2])])
        
        return imgdr

    def __init__(self, inputpath, bands=None, multiband_image=False, bounds=None, buffer = None):
        
        self.boundary = bounds.copy()
        
        if buffer is not None:
            boundary = bounds.copy().buffer(buffer, join_style=2)
        else:
            boundary = bounds.copy()
            
        super().__init__(inputpath, bands, multiband_image, boundary)
        