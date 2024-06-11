import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import Block3DCNN, BlockFC,Downto2D,BlockCNNup,Block,CNNBlock,Unet128
#from .utils import getavgtpool3doutput
from torchvision import models
import torch.nn.functional as nnf
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from typing import List
    
### Pretrained models list
def pre_trainedmodel(modelname:str, input_channels:int, n_outputs:int, perc_params_to_train:float= 0.5):
    """
    Configure a pre-trained model with modifications for a specific number of input channels and outputs.

    Parameters
    ----------
    modelname : str
        The name of the model to configure ('efficientnet', 'transformer', 'densenet', or 'resnet').
    input_channels : int
        The number of input channels for the model.
    n_outputs : int
        The number of outputs for the final layer.
    perc_params_to_train : float, optional
        The percentage of parameters to train (from 0 to 1).

    Returns
    -------
    nn.Module
        The modified pre-trained model.

    Raises
    ------
    ValueError
        If an unsupported model name is provided.
    """
    if modelname == 'efficientnet':
        model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights)
        model_ft.features[0][0] = nn.Conv2d(input_channels, 
                            32, 
                            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, n_outputs)

    
    if modelname == "transformer":
        model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        model_ft.conv_proj = nn.Conv2d(input_channels, 
                            768, 
                            kernel_size=(16, 16), stride=(16, 16))
        


        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, 
                                        n_outputs, bias=True)
                
    if modelname == "densenet":
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.conv0 = nn.Conv2d(input_channels, 
                            64, 
                            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, n_outputs)
        
    if modelname == "resnet":
        model_ft = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model_ft.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
               
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_outputs)
    
    # Set which parameters are trainable based on the percentage provided
    total_params = sum(1 for _ in model_ft.parameters())
    trainable_threshold = int(total_params * (1 - perc_params_to_train))
    
    for i, param in enumerate(model_ft.parameters()):
        param.requires_grad = i > trainable_threshold
        
    return model_ft

class HybridUNetClassifier(nn.Module):
    
    """
    A hybrid model combining U-Net and a pre-trained classification model.
    
    This model uses U-Net as a backbone for feature extraction, and the features are then
    passed to a pre-trained classification model. This setup is suitable for tasks like
    semantic segmentation followed by classification.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels for the U-Net model. Default is 3.
    out_channels : int, optional
        Number of output channels for the U-Net model, which becomes the input channels for the classifier. Default is 3.
    features : int, optional
        Number of features for the U-Net model. Default is 128.
    nlastlayer : int, optional
        Number of output features of the final layer of the classifier. Default is 1.
    classificationmodel : str, optional: 
        The name of the model to configure ('efficientnet', 'transformer', 'densenet', or 'resnet').

    Attributes
    ----------
    unetbb : Unet128
        The U-Net backbone model.
    classifier : nn.Module
        The pre-trained classification model configured for specific inputs and outputs.
    """
    
    def __init__(self,in_channels =3, 
                 out_channels = 3,
                 features = 128, 
                 nlastlayer = 1,
                 classification_model = 'transformer') -> None:
        
        super().__init__()
        
        self.unetbb = Unet128(in_channels=in_channels, out_channels = out_channels, features = features)
        self.model_name = "HybridUnet"+classification_model
        self.resize = torchvision.transforms.Resize((224,224),antialias=True)
        self._resizeflag = classification_model == 'transformer'
        self.classifier = pre_trainedmodel(classification_model, out_channels, nlastlayer)
        

    def forward(self, x):
        d0 = self.unetbb(x)
        if self._resizeflag:
            d0 = self.resize(d0)
            
        d1 = self.classifier(d0)

        return d1
    
    

class HybridCNNClassifier(nn.Module):
    """
    A hybrid model that combines a custom CNN for feature extraction and a pre-trained classification model.
    This architecture is designed for tasks that benefit from specialized feature extraction followed by robust classification.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the CNN model.
    n_lastlayer : int
        Number of output features for the final layer of the classifier.
    features : List[int]
        List specifying the number of features for each convolutional block in the custom CNN.
    blockcnn_dropval : float
        Dropout value to be used in the custom CNN blocks.
    classification_model : str
        Name of the pre-trained model to use ('efficientnet', 'densenet', 'resnet').
    strides : List[int]
        Stride for each convolutional block in the custom CNN.

    Attributes
    ----------
    cnnbb : nn.Module
        Custom CNN blocks used for feature extraction.
    classifier : nn.Module
        Pre-trained classification model configured to process extracted features.
    """
    def __init__(self,in_channels:int =3,
                 n_lastlayer:int = 1,
                 features:List[int] = [128],
                 blockcnn_dropval:float = 0.5,
                 classification_model: str= 'densenet',
                 strides:List[int] = None) -> None:
        
        super().__init__()
        
        self.cnnbb = CNNBlock(in_channels=in_channels, 
                               features=features,strides=strides, blockcnn_dropval =blockcnn_dropval)
        
        self.resize = torchvision.transforms.Resize((224,224),antialias=True)
        self._resizeflag = classification_model == 'transformer'
        self.classifier  = pre_trainedmodel(classification_model, features[len(features)-1], n_lastlayer)
        self.model_name = "HybridCNN"+classification_model

    def forward(self, x):
        d0 = self.cnnbb(x)
        if self._resizeflag:
            d0 = self.resize(d0)
        d1 = self.classifier (d0)
        
        return d1

class ClassDenseNet169(nn.Module):
    
    def __init__(self,in_channels =3, 
                 features = [128],
                 nlastlayer = 1,
                 blockcnn_dropval = 0.5,
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = CNNBlock(in_channels=in_channels, 
                               features=features,strides=strides, blockcnn_dropval =blockcnn_dropval)
        
        self.efficientnet = pre_trainedmodel('efficientnet', features[len(features)-1], nlastlayer)


    def forward(self, x):
        d0 = self.downmt(x)
        
        d1 = self.efficientnet(d0)
        
        return d1
    
    

class RegressionResNetModel(nn.Module):
    
    def __init__(self,in_channels =3, 
                 in_times = 5,
                 features = [128],
                 nlastlayer = 1,
                 block3dconv_dropval = 0.5,
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = Downto2D(in_channels=in_channels,in_times = in_times, 
                               features=features,strides=strides, block3dconv_dropval =block3dconv_dropval)
        
        print(features[len(features)-1])
        
        self.resnet = models.densenet169(pretrained=True)
        self.resnet.features.conv0 = nn.Conv2d(features[len(features)-1], 
                            64, 
                            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        
        for i, param in enumerate(self.resnet.parameters()):
            i 

        nparamstotrain = (i-int((i)*0.5))
        for j, param in enumerate(self.resnet.parameters()):
            if j > nparamstotrain:
                param.requires_grad = True
            else:
                param.requires_grad = False
        num_ftrs = self.resnet.classifier.in_features
        
        self.resnet.classifier = nn.Linear(num_ftrs, nlastlayer)


    def forward(self, x):
        d0 = self.downmt(x)
        
        d1 = self.resnet(d0)
        
        return d1



class RegressionDLModel(nn.Module):
    
    def __init__(self,in_channels =3, 
                 in_times = 5,
                 features = [128],
                 nlastlayer = 1,
                 block3dconv_dropval = 0.5,
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = Downto2D(in_channels=in_channels,in_times = in_times, 
                               features=features,strides=strides, block3dconv_dropval =block3dconv_dropval)
        
        self.imgup1= Block(features[len(features)-1], features[len(features)-1], stride=1, kernel = 35,down=False, act="relu", use_dropout=False)
        self.imgup2 = Block(features[len(features)-1], features[len(features)-1], stride=2, kernel = 36, padding = 1, down=False, act="relu", use_dropout=False)
        
        self.transformer = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.transformer.conv_proj = nn.Conv2d(features[len(features)-1], 
                    768, 
                    kernel_size=(16, 16), stride=(16, 16))
        
        for i, param in enumerate(self.transformer.parameters()):
            i 
        nparamstotrain = (i-int((i)*0.5))
        for j, param in enumerate(self.transformer.parameters()):
            if j > nparamstotrain:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        num_ftrs = self.transformer.heads.head.in_features
        self.transformer.heads.head = nn.Linear(num_ftrs, 
                                nlastlayer, bias=True)

    def forward(self, x):
        d0 = self.downmt(x)
        #out = nnf.interpolate(d0, size=(224, 224), mode='bicubic')
        #d1 = self.imgup(d0)
        d1 = self.imgup1(d0)
        d2 = self.imgup2(d1)
        d3 = self.transformer(d2)
        
        return d3
    

class CNN3DRegression(nn.Module):
    def __init__(self,  in_channels = 3, 
                 in_times = 14,
                 features = [32,64,128,64,32], 
                 widthimg = 84,
                 out_channels=1, fc = [4,8], 
                 use_global = False,
                 block3dconv_dropval = 0.5,
                 fc_dropval = 0.5):
        
        super(CNN3DRegression, self).__init__()
        poolavgkernel = 3
        kernelconv = 3
        kernelavg = 2
        paddingconv = 1
        stridemax = 2
        
        self.use_global = use_global
        
        self.initblock = Block3DCNN(in_channels, features[0],
                                    stride=stridemax,
                                    padding=1)
        
        #disdepths = in_times//stridemax
        layers = []

        in_channels = features[0]

        trimg = (widthimg-1*(kernelconv-1) + 2*paddingconv - 1)//stridemax 
        disdepths = (in_times-1*(kernelconv-1) + 2*paddingconv - 1)//stridemax 
        stopcalcdist = False
        #print(in_channels,trimg, disdepths)
        for i,feature in enumerate(features[1:]):
            
            if disdepths > (poolavgkernel):
                stride = stridemax
                
            else:
                stopcalcdist = True
                stride = 1
                
            layers.append(
                Block3DCNN(in_channels=in_channels, 
                           out_channels = feature, 
                           stride=stride,
                           dropval = block3dconv_dropval)
            )
            trimg =   (trimg-1*(kernelconv-1) + 2*paddingconv - 1)//stride + 1 
            if not stopcalcdist:
                disdepths =   (disdepths-1*(kernelconv-1) + 2*paddingconv - 1)//stride + 1 
            
            in_channels = feature
            #print(feature,trimg, stride, disdepths)
            
        self.conv = nn.Sequential(*layers)
        
        self.convtoft = nn.Sequential(
           nn.AvgPool3d(poolavgkernel, stride=2),
           nn.ReLU(),
           nn.Dropout(block3dconv_dropval)
        )
        trimg = (trimg-poolavgkernel)//kernelavg + 1


        self.flatten = nn.Flatten()
        
        if use_global:
            outconvshape = feature
        else:
            outconvshape = feature*(trimg*trimg) * 1
            
        infc = outconvshape
        layersfc = []
        for con in fc:
            if infc > (con *2):
                layersfc.append(BlockFC(infc,outconvshape//con, 
                                    dropval=fc_dropval))
                infc = outconvshape//con
            
            
        self.fc = nn.Sequential(*layersfc)
        
        
        self.output =  nn.Sequential(
            nn.Linear(infc, out_channels),
            )
             
    
    def forward(self, x):
        
        xinit = self.initblock(x)
        #print(xinit.shape)
        x = self.conv(xinit)
        #print(x.shape)
        x = self.convtoft(x)
        #print(x.shape)
        if self.use_global:
            bs, _, _, _,_ = x.shape
            x = F.adaptive_max_pool3d(x, 1).reshape(bs, -1)
        else:
            x = self.flatten(x)
        
        #print(x.shape)  
        x = self.fc(x)
        #print(x.shape)
        
        x = self.output(x)
        return x


    
    

class ClassificationCNNtransformer(nn.Module):
    
    def __init__(self,in_channels =3, 
                 features = [128],
                 nlastlayer = 1,
                 blockcnn_dropval = 0.5,
                 classification_model = 'transformer',
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = CNNBlock(in_channels=in_channels, 
                               features=features,strides=strides, blockcnn_dropval =blockcnn_dropval)
        
        self.imgup1= Block(features[len(features)-1], features[len(features)-1], stride=1, kernel = 35,down=False,
                           act="relu", use_dropout=False)
        self.imgup2 = Block(features[len(features)-1], features[len(features)-1], stride=2, kernel = 38, padding = 1, 
                            down=False, act="relu", use_dropout=False)
        
        self.transformer = pre_trainedmodel(classification_model, features[len(features)-1], nlastlayer)
        
        self.model_name = "CNN"+classification_model
        
    def forward(self, x):
        d0 = self.downmt(x)
        
        d1 = self.imgup1(d0)
        d2 = self.imgup2(d1)
        d3 = self.transformer(d2)

        return d3


class Transformer(nn.Module):
    def __init__(self,in_channels = 3, noutputs=1):
        
        super().__init__()
        self.transformer = pre_trainedmodel('transformer', in_channels, noutputs)
        
    
    def forward(self, x):
        x = self.transformer(x)
       
        return x
    

class Classification3DCNNEfficientNet(nn.Module):
    
    def __init__(self,in_channels =3, 
                 in_times = 5,
                 features = [128],
                 nlastlayer = 1,
                 block3dconv_dropval = 0.5,
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = Downto2D(in_channels=in_channels,in_times = in_times, 
                               features=features,strides=strides, block3dconv_dropval =block3dconv_dropval)
        
        self.efficientnet = pre_trainedmodel('efficientnet', features[len(features)-1], nlastlayer)


    def forward(self, x):
        d0 = self.downmt(x)
        
        d1 = self.efficientnet(d0)
        
        return d1
    
class Classification3DCNNtransformer(nn.Module):
    
    def __init__(self,in_channels =3, 
                 in_times = 5,
                 features = [128],
                 nlastlayer = 1,
                 block3dconv_dropval = 0.5,
                 strides = None) -> None:
        
        super().__init__()
        
        self.downmt = Downto2D(in_channels=in_channels,in_times = in_times, 
                               features=features,strides=strides, block3dconv_dropval =block3dconv_dropval)
        
        
        self.imgup1= Block(features[len(features)-1], features[len(features)-1], stride=1, kernel = 35,down=False,
                           act="relu", use_dropout=False)
        self.imgup2 = Block(features[len(features)-1], features[len(features)-1], stride=2, kernel = 36, padding = 1, 
                            down=False, act="relu", use_dropout=False)
        
        self.transformer = pre_trainedmodel('transformer', features[len(features)-1], nlastlayer)
        
        self.relu = nn.ReLU()
        

    def forward(self, x):
        d0 = self.downmt(x)
        #out = nnf.interpolate(d0, size=(224, 224), mode='bicubic')
        #d1 = self.imgup(d0)
        d1 = self.imgup1(d0)
        d2 = self.imgup2(d1)
        d3 = self.transformer(d2)
        #d2 = self.relu(d1)
        #d2 = self.relu(d1)
        return d3
    
    
class TransformerMLTC(nn.Module):
    
    def __init__(self,in_channels =3, 
                 nlastlayer = 1,
                 ) -> None:
        
        super().__init__()
        
        self.transformer = pre_trainedmodel('transformer', in_channels, nlastlayer)
        

    def forward(self, x):
        d1 = self.transformer(x)
        #d2 = self.relu(d1)
        #d2 = self.relu(d1)
        return d1
    

##### INSTANCE SEGMENTATION MODELS


def maskrcnn_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class SegmAny:
    def __init__(self, model, device = 'cuda') -> None:
        from segment_anything import SamPredictor
        
        self.model = model
        self.model.to(device)
        self.predictor = SamPredictor(self.model)
        self.image = None
        
    def set_image(self, image):
        
        self.predictor.set_image(image)
        self.image = image
        
    def predict_coord(self, x, y, label = None):
        assert self.image is not None #("set image first")
        
        input_point = np.array([[x, y]])
        if label is None:
            input_label = np.array([1])
            
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        #mask_input = logits[np.argmax(scores), :, :] 
        return masks[np.argmax(scores)]