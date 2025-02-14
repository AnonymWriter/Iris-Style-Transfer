import torch

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