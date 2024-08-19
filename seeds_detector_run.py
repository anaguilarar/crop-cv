from cropdatacube.datacubepredictors.segmentation import DLInstanceModel
from cropdatacube.datasets.data_preprocess import PreProcess_InstaData
from cropdatacube.phenotyping.counter import SeedsCounter
from cropdatacube.cropcv.readers import ImageReader
from cropdatacube.utils.general import FolderWithImages
from cropdatacube.ml_utils.models.available import check_weigth_path
from cropdatacube.phenotyping.calculate_seeds_metrics import batch_seed_image_detection

from PIL import Image
from omegaconf import OmegaConf

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Seed High-Throughput Phenotyping')
    parser.add_argument('--config', 
                        default='examples/seed_detection_configuration.yaml', help='config file path')

    args = parser.parse_args()
    return args


def main():
    logging.info("Starting Seed High-Throughput Phenotyping")
    args = parse_args()
    
    # reading configuration
    config = OmegaConf.load(args.config)
    
    # setting model    
    modelsweight_path = check_weigth_path(config.MODEL.modelweigthurl, suffix=config.MODEL.model_output_name, weights_path="")
    logging.info(f"Model saved in {modelsweight_path}")
    
    seg_model = DLInstanceModel(
        os.path.join("",config.MODEL.model_output_folder_name, config.MODEL.model_output_name))

    # setting imagery transformation
    transformopts = PreProcess_InstaData(mean_scaler_values= config.DATASET.mean_scaler_values,
                                        std_scaler_values= config.DATASET.std_scaler_values)

    
    # Seed detection module
    seeds_analizer = SeedsCounter(
                            detector=seg_model.model,
                            device=seg_model.device,
                            detector_size = (config.MODEL.inputsize,config.MODEL.inputsize),
                            transform=transformopts)
    
    # Reading image paths in folder
    inputpath = config.DATASET.image_path
    imagespath = FolderWithImages(path= inputpath, suffix=config.DATASET.images_suffix)
    
    logging.info(f"files available in folder: {len(imagespath.files_in_folder)}")
    
    # setting output folder
    
    results_output = config.OUTPUT.path
    ## create output folder
    if not os.path.exists(results_output):
        os.mkdir(results_output)

    logging.info(f"Detection results will be saved in {results_output}")
    
    ## run models
    
    alldata = batch_seed_image_detection( imagespath.files_in_folder, config, seeds_analizer, results_output, images_suffix = config.DATASET.images_suffix)
    pd.concat(alldata).to_csv(os.path.join(results_output,'alldata.csv'))
    
if __name__ == "__main__":
    main()