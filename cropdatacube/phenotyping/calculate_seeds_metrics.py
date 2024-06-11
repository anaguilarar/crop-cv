from ..cropcv.readers import ImageReader
from tqdm import tqdm
import os
from PIL import Image

def batch_seed_image_detection(image_paths, configuration, seed_detector, outputpath, images_suffix = '.jpg'):
    
    alldata = []
    for i in tqdm(range(len(image_paths))):
        
        fn = image_paths[i]
        imgr = ImageReader()
        img = imgr.get_image(path = fn)
        try:
            
            ## output files
            ## file name
            outputplotfn = imgr._split_filename(fn)[-1]
            if not outputplotfn.endswith('.png'):
                outputplotpath = outputplotfn[:outputplotfn.index(images_suffix)]+'.png'
                outputtablepath = outputplotfn[:outputplotfn.index(images_suffix)]
                
            ## detection
            seed_detector.detect_seeds(image=img,prediction_threshold=configuration.MODEL.prediction_threshold, 
                                segmentation_threshold = configuration.MODEL.segmentation_threshold )

            if not len(seed_detector.bbs) > 1:
                continue
            ## all seeds plot
            seedsimgpath = os.path.join(outputpath,'detected_seeds_'+outputplotpath)
            ## show detections
            imgseeds = seed_detector.visualize_detected_seeds(label_factorsize = configuration.PLOTS.label_factorsize, 
                                                            heightframefactor = configuration.PLOTS.heightframefactor, 
                                                            textthickness=4, 
                                                            widthframefactor = configuration.PLOTS.widthframefactor)
            
            Image.fromarray(imgseeds).save(seedsimgpath)

            ## individual seeds
            seedsmetricspath = os.path.join(outputpath,'seed_metrics_'+outputplotpath)
            
            f = seed_detector.plot_all_seeds_metrics(ncols = 4,perpendicular_tolerance=configuration.PROCESS.perpendicular_tolerance, 
                                                    padding_percentage=configuration.PROCESS.padding_percentage, export_path = seedsmetricspath, figsize = configuration.PLOTS.seeds_figsize)
            
            # export metrics
            dftoexport = seed_detector.get_all_seed_metrics(perpendicular_tolerance=configuration.PROCESS.perpendicular_tolerance,
                                            padding_percentage=configuration.PROCESS.padding_percentage, 
                                            color_spacelist= configuration.PROCESS.color_spacelist, include_srgb = configuration.PROCESS.include_srgb, 
                                            quantiles = configuration.PROCESS.color_quantiles)
            
            dftoexport['image'] = outputtablepath
            seedstablepath = os.path.join(outputpath,'seed_metrics_'+outputtablepath+'.csv')
            dftoexport.to_csv(seedstablepath)
            alldata.append(dftoexport)
        except:
            raise Exception(f'Error with file {fn}')
        
    return alldata