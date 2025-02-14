import torch
import torchvision.transforms.v2 as transforms
from torchvision.models import resnet50, ResNet50_Weights
            
class ResNet50(torch.nn.Module):
    """
    A shell to pretrained pytorch ResNet50 model.
    """
    
    def __init__(self, freeze: bool = True) -> None:
        """
        Arguments:
            freeze (bool): whether to freeze the model.
        """
        
        super().__init__()
        
        self.model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        
        # drop last layer
        self.model.fc = torch.nn.Identity()
        
        # put model in eval mode and freeze model
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # transform
        self.t = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale = True),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): input images or image tensors.

        Returns:
            x (torch.Tensor): features.
        """

        x = self.t(x)
        
        # expect x of shape (b, c, h, w)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
                
        x = self.model(x)

        return x