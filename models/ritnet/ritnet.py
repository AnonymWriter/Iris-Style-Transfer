import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
    
class RITnet(torch.nn.Module):
    """
    A shell to pre-trained RITnet model.
    """
    
    def __init__(self, 
                 dropout: bool = True, 
                 prob: float = 0.2, 
                 load_pretrained: bool = True, 
                 pretrained_path: str = 'models/weights/ritnet_pretrained.pkl'
                 ) -> None:
        """
        Arguments:
            dropout (bool): whether to use dropout.
            prob (float): dropout probability.
            load_pretrained (bool): whether to load the pre-trained parameters.
            pretrained_path (str): path to pre-trained model file.
        """
        
        super().__init__()
        self.model = DenseNet2D(dropout = dropout, prob = prob)
        if load_pretrained:
            self.model.load_state_dict(torch.load(pretrained_path, weights_only = True, map_location = 'cpu'))
        
        # put model in eval mode and freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # transform
        self.t = RITnet_transform()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
        
        Returns:
            x (torch.Tensor): predicted segmentation labels.
        """

        with torch.no_grad():
            x = self.t(x)
            x = self.model(x)

            # postprocessing
            b, c, h, w = x.size()
            _, x = x.max(1)
            x = x.view(b, h, w)
        return x

##################################### resource #####################################
# source 1: UNet, https://github.com/ShusilDangi/DenseUNet-K
# source 2: RITnet, https://bitbucket.org/eye-ush/ritnet/src/master/
####################################################################################

class RITnet_transform(torch.nn.Module):
    """
    RITnet transformations.
    """
    
    def __init__(self):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit = 1.5, tileGridSize = (8, 8))
        self.table = 255.0 * (np.linspace(0, 1, 256)**0.8)
        self.transform = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale = True),
            transforms.Normalize([0.5], [0.5])
            ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments: 
            x (torch.Tensor): input image tensor.

        Returns:
            x (torch.Tensor): transformed input image tensor.
        """
        # x = x.convert("L")
        if len(x.shape) == 3: # image of shape (1, h, w)
            x = x[0]
        device = x.device
        x = x.to('cpu')
        x = (x * 255).to(torch.uint8)
        x = cv2.LUT(np.array(x), self.table) # cv2.LUT(np.array(x), self.table)
        x = self.clahe.apply(np.array(np.uint8(x)))
        # x = Image.fromarray(x)
        x = self.transform(x)
        x = x.unsqueeze(0).to(device)        
        return x

class DenseNet2D_down_block(torch.nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = torch.nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = torch.nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = torch.nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = torch.nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = torch.nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = torch.nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=prob)
        self.dropout2 = torch.nn.Dropout(p=prob)
        self.dropout3 = torch.nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)
    
class DenseNet2D_up_block_concat(torch.nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = torch.nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = torch.nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = torch.nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = torch.nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = torch.nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=prob)
        self.dropout2 = torch.nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = F.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out    
class DenseNet2D(torch.nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=32,concat=True,dropout=False,prob=0):
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = torch.nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = torch.nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out