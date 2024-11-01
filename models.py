import cv2
import tqdm
import math
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from torchvision.models import vgg19, vgg19_bn, VGG19_Weights, VGG19_BN_Weights

# self-defined functions
from utils import GramMatrix, crop_image, cal_mIoU, sample_other

# vgg structures (without batch normalization layers)
vgg19_layers = {'conv1_1': 0,  'relu1_1': 1,  'conv1_2': 2,  'relu1_2': 3,  'pool1': 4,
                'conv2_1': 5,  'relu2_1': 6,  'conv2_2': 7,  'relu2_2': 8,  'pool2': 9,
                'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13, 'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
                'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22, 'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
                'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31, 'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36}

# vgg structures (with batch normalization layers)
vgg19_bn_layers = {'conv1_1': 0,  'bn1_1': 1,  'relu1_1': 2,  'conv1_2': 3,  'bn1_2': 4,  'relu1_2': 5,  'pool1': 6,
                   'conv2_1': 7,  'bn2_1': 8,  'relu2_1': 9,  'conv2_2': 10, 'bn2_2': 11, 'relu2_2': 12, 'pool2': 13,
                   'conv3_1': 14, 'bn3_1': 15, 'relu3_1': 16, 'conv3_2': 17, 'bn3_2': 18, 'relu3_2': 19, 'conv3_3': 20, 'bn3_3': 21,'relu3_3': 22, 'conv3_4': 23, 'bn3_4': 24,'relu3_4': 25, 'pool3': 26,
                   'conv4_1': 27, 'bn4_1': 28, 'relu4_1': 29, 'conv4_2': 30, 'bn4_2': 31, 'relu4_2': 32, 'conv4_3': 33, 'bn4_3': 34,'relu4_3': 35, 'conv4_4': 36, 'bn4_4': 37,'relu4_4': 38, 'pool4': 39,
                   'conv5_1': 40, 'bn5_1': 41, 'relu5_1': 42, 'conv5_2': 43, 'bn5_2': 44, 'relu5_2': 45, 'conv5_3': 46, 'bn5_3': 47,'relu5_3': 48, 'conv5_4': 49, 'bn5_4': 50,'relu5_4': 51, 'pool5': 52}

class VGG19(torch.nn.Module):
    """
    A shell to pretrained pytorch VGG19 model.
    """
    
    def __init__(self, 
                 content_layers: list[str] = ['relu4_2'], 
                 style_layers: list[str] = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], 
                 bn: bool = False,
                 ) -> None:
        """
        Arguments:
            content_layers (list[str]): layers for content feature extraction.
            style_layer (list[str]): layers for style feature extraction.
            bn (bool): whether to use vgg model with batch normalization layers or not.
        """
        
        super().__init__()

        if bn:
            self.model = vgg19_bn(weights = VGG19_BN_Weights.IMAGENET1K_V1)
            self.content_layers_idx = [vgg19_bn_layers[i] for i in content_layers]
            self.style_layers_idx = [vgg19_bn_layers[i] for i in style_layers]
        else:
            self.model = vgg19(weights = VGG19_Weights.IMAGENET1K_V1)
            self.content_layers_idx = [vgg19_layers[i] for i in content_layers]
            self.style_layers_idx = [vgg19_layers[i] for i in style_layers]

        # drop projection head (the fully connected linear layers in the end)
        self.model = self.model.features

        # put model in eval mode and freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # determine feature layers
        feature_layers_idx = list(set(self.content_layers_idx) | set(self.style_layers_idx))
        
        # wrap up content and style layers with feature extractors
        for i in feature_layers_idx:
            self.model[i] = FeatureExtractor(self.model[i])
    
        # normalization
        self.normalize = transforms.Normalize(mean = torch.tensor([0.485, 0.456, 0.406]), std = torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensors.
            mask (torch.Tensor): input mask tensors.

        Returns:
            x (torch.Tensor): CNN features.
            content_features (list[torch.Tensor]): content features.
            style_features (list[torch.Tensor]): style features.
        """

        x = self.normalize(x)

        # apply mask
        if mask is not None:
            x = x * mask

        x = self.model(x)

        content_features = [self.model[i].feature for i in self.content_layers_idx]
        style_features = [self.model[i].feature for i in self.style_layers_idx]

        return x, content_features, style_features
    
class FeatureExtractor(torch.nn.Module):
    """
    A layer that saves feature on feed-forward.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        """
        Arguments:
            layer (torch.nn.Module): pytorch layer.
        """

        super().__init__()
        self.layer = layer
        self.feature = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): identical as input x.
        """
        x = self.layer(x)
        self.feature = x
        return x
    
class Classifier1(torch.nn.Module):
    """
    Projection head for CNN features.
    """

    def __init__(self, num_class: int = 152):
        """
        Arguments:
            num_class (int): number of unique classes.
        """

        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size = (7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 25088, out_features = 4096, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = 4096, out_features = num_class, bias = True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): CNN features.

        Returns:
            (torch.Tensor): logits (not softmaxed yet).
        """

        return self.model(x)
    
class Classifier2(torch.nn.Module):
    """
    Projection head for style features.
    """

    def __init__(self, in_features: int = (64 + 128 + 256 + 512) * 2, num_class: int = 152):
        """
        Arguments:
            in_features (int): style feature dimension.
            num_class (int): number of unique classes.
        """
        
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_features, out_features = 4096, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = 4096, out_features = num_class, bias = True)
        )

    def forward(self, style_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            style_features (list[torch.Tensor]): style features.

        Returns:
            (torch.Tensor): logits (not softmaxed yet).
        """
        
        # compute style features in form of BN statistics (mean and std)
        style_features = torch.cat([torch.cat([x.mean(dim = (-2, -1)), x.std(dim = (-2, -1))], dim = 1) for x in style_features], dim = 1)
        return self.model(style_features)

class ContentLoss_L2(torch.nn.Module):
    """
    Loss function for MSE-based content loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target content features.
            weights (list[float]): weight for each layer.
        """
        
        super().__init__()
        self.targets = targets
        self.weights = [1.0] * len(targets) if weights is None else weights
        # self.weights = [w / sum(self.weights) for w in weights]   # normalize weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): content features of input image.

        Returns:
            loss (torch.Tensor): content loss.
        """
        
        loss = 0
        for p, t, w in zip(preds, self.targets, self.weights):
            # loss += ((p - t)**2).sum() * w
            loss += F.mse_loss(p, t) * w
        loss *= 0.5
        return loss
    
class StyleLoss_Gram(torch.nn.Module):
    """
    Loss function for MSE-based style loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target style features.
            weights (list[float]): weight for each layer.
        """
         
        super().__init__()
        self.targets = [GramMatrix(t) for t in targets]
        self.weights = [1.0] * len(targets) if weights is None else weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): style features of input image.

        Returns:
            loss (torch.Tensor): style loss.
        """

        loss = 0
        for p, t, w in zip(preds, self.targets, self.weights):
            p = GramMatrix(p)
            loss += ((p - t)**2).sum() * w
        loss *= 0.25
        return loss

class StyleLoss_BN(torch.nn.Module):
    """
    Loss function for BN statistics-based style loss.
    """
    
    def __init__(self, targets: list[torch.Tensor] = None, weights: list[float] = None) -> None:
        """
        Arguments:
            targets (list[torch.Tensor]): target style features.
            weights (list[float]): weight for each layer.
        """

        super().__init__()
        self.targets_mean = [t.mean(dim = (-2, -1)) for t in targets]
        self.targets_std = [t.std(dim = (-2, -1)) for t in targets]
        self.weights = [1.0] * len(targets) if weights is None else weights
        
    def forward(self, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        Arguments:
            preds (list[torch.Tensor]): style features of input image.

        Returns:
            loss (torch.Tensor): style loss.
        """

        loss = 0
        for p, t_mean, t_std, w in zip(preds, self.targets_mean, self.targets_std, self.weights):
            p_mean = p.mean(dim = (-2, -1))
            p_std = p.std(dim = (-2, -1))
            loss += ((p_mean - t_mean)**2 + (p_std - t_std)**2).sum() * w / p_mean.shape[-1]
        return loss

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
    
class RITnet(torch.nn.Module):
    """
    A shell to pre-trained RITnet model.
    """
    
    def __init__(self, 
                 dropout: bool = True, 
                 prob: float = 0.2, 
                 load_pretrained: bool = True, 
                 pretrained_path: str = './models/ritnet_pretrained.pkl'
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

        x = self.t(x)
        x = self.model(x)

        # postprocessing
        b, c, h, w = x.size()
        _, x = x.max(1)
        x = x.view(b, h, w)
        return x
    
class IrisDataset(torch.utils.data.Dataset):
    """
    Self-defined dataset, used for iris classification.
    """
    
    def __init__(self, 
                 xs: list[torch.Tensor], 
                 ys: list[int], 
                 rotation_prob: float = 0,
                 rotation_degree: float = 180,
                 perspect_prob: float = 0,
                 perspect_degree: float = 0.3,
                 glint_threshold: float = 0.8,
                 ritnet: torch.nn.Module = None,
                 device: str = 'cuda:0',
                 ) -> None:
        """
        Arguments:
            xs (list[torch.Tensor]): image tensors.
            ys (list[int]): class labels.
            rotation_prob (float): probability of random rotation.
            rotation_degree (float): degree of random rotation.
            perspect_prob (float): probability of random perspective transformation.
            perspect_degree (float): degree of random perspective transformation.
            glint_threshold (float): threshold for glints.
            ritnet (torch.nn.Module): RITnet model.
            device (str): CPU or GPU.
        """
        
        assert(len(xs) == len(ys))

        self.xs = []
        self.ys = torch.as_tensor(ys).long().to(device)
        
        # ritnet model
        if ritnet is None:
            ritnet = RITnet()
        ritnet.to(device)

        # transforms
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p = perturb_prob),
            # transforms.RandomVerticalFlip(p = perturb_prob),
            transforms.RandomApply([transforms.RandomRotation(degrees = rotation_degree)], p = rotation_prob),
            transforms.RandomPerspective(distortion_scale = perspect_degree, p = perspect_prob)
            ])
        
        print('processing data...')
        for x in tqdm.tqdm(xs):
            x = x.to(device)
            
            # compute ritnet mask
            m_ritnet = ritnet(x)
            m_ritnet = m_ritnet == 2 # 2 represents iris part

            # compute non-glint mask
            m_glint = x <= glint_threshold

            # apply both masks
            x = x * m_ritnet * m_glint

            # crop image
            x = crop_image(x)

            # transforms
            x = t(x)

            self.xs.append(x)
             
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """

        return len(self.ys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Arguments:
            idx (int): index.

        Returns:
            (torch.Tensor): image tensor.
            (torch.Tensor): class label.
        """
        return self.xs[idx], self.ys[idx]
    
class EyeNSTDataset(torch.utils.data.Dataset):
    """
    Self-defined dataset, used for iris style transfer.
    """
    
    def __init__(self, 
                 c_imgs: list[torch.Tensor], 
                 c_labels: list[int],
                 c_masks_gt: list[torch.Tensor],
                 ritnet: torch.nn.Module = None,
                 device: str = 'cuda:0',
                 glint_threshold: float = 0.8
                 ) -> None:
        """
        Arguments:
            c_imgs (list[torch.Tensor]): content image tensors. 
            c_labels (list[int]): content image class labels.
            c_masks_gt (list[torch.Tensor]): content image segmentation ground truth labels.
            ritnet (torch.nn.Module): RITnet model.
            device (str): CPU or GPU.
            glint_threshold (float): threshold for glints.
        """
        
        assert(len(c_imgs) == len(c_labels) == len(c_masks_gt))
        
        # regarding content images
        self.c_imgs = []
        self.c_labels = torch.as_tensor(c_labels).long().to(device)
        self.c_masks_gt = torch.stack(c_masks_gt).to(device)
        self.c_masks_iris = []
        self.c_iris_bbs = []
        self.mious = []
        
        # regarding style images
        self.s_irises = []
        self.s_labels = []
                
        # ritnet model
        if ritnet is None:
            ritnet = RITnet()
        ritnet.to(device)
        
        # transform
        t = transforms.Resize((224, 224))
        
        print('processing data...')
        for c_img, c_label, c_m_gt in tqdm.tqdm(zip(c_imgs, self.c_labels, self.c_masks_gt), total = len(c_labels)):
            c_img = c_img.to(device)
            self.c_imgs.append(c_img)
            
            # compute content image ritnet mask
            c_m_ritnet = ritnet(c_img)
            
            # compute mIoU
            miou = cal_mIoU(c_m_ritnet, c_m_gt.unsqueeze(0))
            self.mious.append(miou)

            # compute content image non-glint mask
            c_m_glint = c_img <= glint_threshold

            # apply both masks and compute bounding box
            c_m_ritnet = c_m_ritnet == 2
            c_m = c_m_ritnet * c_m_glint
            self.c_masks_iris.append(c_m)
            c_img = c_img * c_m
            c_x_min, c_y_min, c_x_max, c_y_max = crop_image(c_img, return_idx = True)
            self.c_iris_bbs.append(torch.tensor([c_x_min, c_y_min, c_x_max, c_y_max]))
            
            # randomly sample a style image, which is an eye image of another user
            s_idx = sample_other(c_label, c_labels)
            self.s_labels.append(self.c_labels[s_idx])
            s_img = c_imgs[s_idx].to(device)
            
            # for style image, we only need the iris part
            s_m_ritnet = ritnet(s_img)
            s_m_ritnet = s_m_ritnet == 2
            s_m_glint = s_img <= glint_threshold
            s_img = s_img * s_m_ritnet * s_m_glint
            s_iris = crop_image(s_img)
            s_iris = t(s_iris)
            self.s_irises.append(s_iris)
             
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """
        return len(self.c_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor,              # content image tensor
                                             int,                       # content image class label
                                             torch.Tensor,              # content image iris mask
                                             tuple[int, int, int, int], # content image iris bounding box corners
                                             torch.Tensor,              # content image ground truth segmentation labels
                                             torch.Tensor,              # style image tensor
                                             int]:                      # style image class label
        """
        Arguments:
            idx (int): index.

        Returns:
        (torch.Tensor): content image tensor
        (int): content image class label
        (torch.Tensor): content image iris mask
        (tuple[int, int, int, int]): content image iris bounding box corners
        (torch.Tensor): content image ground truth segmentation labels
        (torch.Tensor): style image tensor
        (int): style image class label
        """
        
        return self.c_imgs[idx],        \
               self.c_labels[idx],      \
               self.c_masks_iris[idx],  \
               self.c_iris_bbs[idx],    \
               self.c_masks_gt[idx],    \
               self.s_irises[idx],      \
               self.s_labels[idx]

##################################### RITnet #####################################
# source 1: UNet, https://github.com/ShusilDangi/DenseUNet-K
# source 2: RITnet, https://bitbucket.org/eye-ush/ritnet/src/master/
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