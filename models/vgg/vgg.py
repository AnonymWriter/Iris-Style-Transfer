import torch
import torchvision.transforms.v2 as transforms
from torchvision.models import vgg19, vgg19_bn, VGG19_Weights, VGG19_BN_Weights

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
    
        # transform
        self.t = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale = True),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

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

        x = self.t(x)

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