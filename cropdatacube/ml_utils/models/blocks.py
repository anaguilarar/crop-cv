import torch.nn as nn
import torch
import torch.nn.functional as F

def calculateconvtranspose(hin,stride,padding,kernel,output_padding=0,dilation=1):
    return (hin - 1) * stride - 2 * padding+dilation* (kernel - 1) + output_padding+1

def calckernelconvtranspose(hin,houtput,stride,padding,output_padding=0,dilation=1):
    kernel = ((houtput - (hin - 1) * stride + 2 * padding + dilation)/ dilation ) - 1 
    return kernel


def getconv3d_dimsize(din,kernel_size,stride,padding,dilation = 1):
    return ((din +2  * padding - dilation * (kernel_size-1) - 1)//stride) + 1
def getavgtpool3doutput(din,kernel_size,stride,padding):
    return ((din + 2*padding - kernel_size) // stride) + 1 

### taken from https://holmdk.github.io/2020/04/02/video_prediction.html
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)



class BlockCNNup(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel = 3,
                 stride=1, padding = 0,use_dropout = True,
                 dropval = 0.5) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride = stride , padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropval)
        self.use_dropout = use_dropout
        
    def forward(self, x):
        x = self.conv(x)
        
        return self.dropout(x) if self.use_dropout else x
    

class Block3DCNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel = 3,
                 stride=2, padding = 1,use_dropout = True,
                 dropval = 0.5) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, stride,padding, 
                      bias=False, padding_mode="reflect"),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropval)
        self.use_dropout = use_dropout
        
    def forward(self, x):
        x = self.conv(x)
        
        return self.dropout(x) if self.use_dropout else x

     
class BlockFC(nn.Module):
    def __init__(self, fc, fcout, use_dropout = True, use_batch = True, dropval = 0.5) -> None:
        super().__init__()
        self.fc_layer = nn.Linear(fc, fcout)
        self.batch = nn.BatchNorm1d(fcout)
        self.dropout = nn.Dropout(dropval)
        self.use_dropout = use_dropout
        self.use_batch = use_batch
        
        
    def forward(self, x):
        
        x = self.fc_layer(x)
        if self.use_batch:
            x = self.batch(x)
            
        return F.relu(self.dropout(x)) if self.use_dropout else F.relu(x)
        
class Downto2D(nn.Module):
    def __init__(self, in_channels=3, in_times = 14,features=32, strides = None,block3dconv_dropval = 0.3):
        
        super().__init__()
        features = features if type(features) is list else [features]
        if strides is None:
            strides = [(1,1,1)]
            strides = strides + [(2,1,1)]*len(features)
        
        ndim = in_times
        for i in range(len(features)):
            ndim = getconv3d_dimsize(ndim,kernel_size = 3,
                                    stride = strides[i][0],
                                    padding = 1,dilation = 1)
            if ndim == 1 and i<(len(features)-1):
                strides[i+1] = (1,1,1)

            #
        
        self.initblock = Block3DCNN(in_channels, features[0],
                                    stride=strides[0])
        
        layers = []

        in_channels = features[0]
        
        for i,feature in enumerate(features[1:]):
            
                
            layers.append(
                Block3DCNN(in_channels=in_channels, 
                           out_channels = feature, 
                           stride=strides[i+1],
                           dropval = block3dconv_dropval)
            )
            in_channels = feature
        
        self.conv = nn.Sequential(*layers)
        
        self.convtoft = nn.Sequential(
           nn.AvgPool3d((ndim,3,3), stride=1, padding=(0,1,1)),
           nn.ReLU(),
           nn.Dropout(block3dconv_dropval)
        )

    def forward(self, x):

        x = self.initblock(x)

        x = self.conv(x)

        if x.shape[2]>1:
            #print('convavg3d')
            #print(x.shape)
            x = self.convtoft(x)
            #print(x.shape)
        resh = torch.reshape(x, (x.shape[0],x.shape[1],x.shape[3],x.shape[4]))

        return resh    

class Downto2D128(nn.Module):
    def __init__(self, in_channels=3, features=16):
        super().__init__()
        self.down1 = Block3DCNN(in_channels, features*4, stride=(2,1,1))
        self.down2 = Block3DCNN(features*4,features*4, stride=(2,1,1))
        self.down3 = Block3DCNN(features*4,features*2, stride=(2,2,2))
        
        ## self reshape
        #self.reshapet = torch.reshape((in_channels,h,w))
    
    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        

        resh = torch.reshape(d3, (d3.shape[0],d3.shape[1],d3.shape[3],d3.shape[4]))
        return resh  
    


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 4, stride=2, 
                 padding = 1, down=True, act="relu", use_dropout=False, dropval = 0.5):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropval)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class CNNBlock(nn.Module):
    def __init__(self, in_channels=3,features=32, strides = None,blockcnn_dropval = 0.5):
        
        super().__init__()
        features = features if type(features) is list else [features]
        if strides is None:
            strides = [(1,1)]
            strides = strides + [(1,1)]*len(features)
        

        self.initblock = Block(in_channels, features[0],
                                    stride=strides[0])
        
        layers = []

        in_channels = features[0]
        
        for i,feature in enumerate(features[1:]):
            
                
            layers.append(
                Block(in_channels=in_channels, 
                           out_channels = feature, 
                           stride=strides[i+1],
                           dropval = blockcnn_dropval)
            )
            in_channels = feature
        
        self.conv = nn.Sequential(*layers)
        


    def forward(self, x):
        x = self.initblock(x)
        x = self.conv(x)
        return x    





class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2, kernel = 4, padding = 1, down=True, act="relu", use_dropout=False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class Unet128(nn.Module):
    
    def __init__(self, in_channels = 3, out_channels=3, features=64):
        super(Unet128,self).__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features*2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 4 * 2, features *2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features *2*2, features , down=False, act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):

        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        #print(d6.shape)
        bottleneck = self.bottleneck(d6)

        up1 = self.up1(bottleneck)
        #print('up1',up1.shape, d6.shape)
        up2 = self.up2(torch.cat([up1, d6], 1))
        #print('up2',up2.shape, d5.shape)
        up3 = self.up3(torch.cat([up2, d5], 1))
        #print('up3',up3.shape, d4.shape)
        up4 = self.up4(torch.cat([up3, d4], 1))
        #print('up4',up4.shape, d3.shape)
        up5 = self.up5(torch.cat([up4, d3], 1))
        #print('up5',up5.shape, d2.shape)
        up6 = self.up6(torch.cat([up5, d2], 1))
        #print('up6',up6.shape, d1.shape)
        upfinal  = self.final_up(torch.cat([up6, d1], 1))
        
        return upfinal    
    