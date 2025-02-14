# resource: https://github.com/zeenolife/openeds_2020

import torch
import segmentation_models_pytorch as smp
import torchvision.transforms.v2 as transforms

class EfficientNet(torch.nn.Module):
    """
    A shell to pre-trained EfficientNet model from segmentation_models_pytorch.
    """
    
    def __init__(self, load_pretrained: bool = True, pretrained_path: str = './models/weights/unet_efficientnet-b7.pt') -> None:
        """
        Arguments:
            load_pretrained (bool): whether to load the pre-trained parameters.
            pretrained_path (str): path to pre-trained model file.
        """
        
        super().__init__()
        self.model = smp.Unet(encoder_name = 'efficientnet-b7', classes = 4)
        if load_pretrained:
            paras = torch.load(pretrained_path, weights_only = False, map_location = "cpu")
            paras = {k[7:] : v  for k, v in paras['state_dict'].items()} # skip 'module.' in key
            self.model.load_state_dict(paras)
        
        # put model in eval mode and freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # transform
        self.t = transforms.Compose([transforms.ToImage(), 
                                     transforms.ToDtype(torch.float32, scale = True),
                                     transforms.Pad(padding = (0, 8, 0, 8), fill = 0, padding_mode = 'constant'),
                                     transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                     ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input image tensor.
            
        Returns:
            o (torch.Tensor): predicted segmentation labels.
        """
        
        with torch.no_grad():
            x = self.t(x)
            
            # expect x of shape (b, c, h, w)
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
                
            o1 = self.model(x)
            
            # test-time augmentation
            o2 = torch.flip(self.model(torch.flip(x, dims=(3,))), dims=(3,)) 
            o = (o1 + o2) / 2
            
            # get segmentation labels
            o = torch.softmax(o, dim = 1)
            o = torch.argmax(o, dim = 1)
            
            # crop the padded region
            o = o[:, 8:-8, :]
            
        return o