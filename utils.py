import os
import torch
import random
import shutil
import skimage
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score

# metrics that require average parameter
metrics_with_avg = {'prec' : precision_score, 'recl' : recall_score, 'f1' : f1_score}
avg = 'macro'

# metrics that dont require average parameter
metrics_no_avg = {'accu' : accuracy_score, 'mcc' : matthews_corrcoef}

def seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def prepare_dir(dir: str) -> None:
    """
    Prepare the directory for savings.

    Arguments:
        dir (str): directory path.
    """

    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def crop_image(image: torch.Tensor, return_idx: bool = False) -> torch.Tensor | tuple[int, int, int, int]:
    """
    Trim the black border of an image.
    
    Arguments:
        image (torch.Tensor): image tensor.
        return_idx (bool): whether to return trimmed image, or the trimming bounding box.

    Returns:
        x_min, y_min, x_max, y_max (tuple[int, int, int, int]): trimming bounding box.
        cropped (torch.Tensor): trimmed image tensor.
    """

    nonzero = image.nonzero()

    if len(image.shape) == 2: # image of shape (h, w)
        x_min, y_min = nonzero.min(dim = 0)[0]
        x_max, y_max = nonzero.max(dim = 0)[0]
    elif len(image.shape) == 3 and image.shape[0] == 1: # image of shape (1, h, w)
        _, x_min, y_min = nonzero.min(dim = 0)[0]
        _, x_max, y_max = nonzero.max(dim = 0)[0]
    else:
        raise Exception('image shape wrong:', image.shape)

    if return_idx:
        return x_min, y_min, x_max, y_max
    else:
        cropped = image[:, x_min: x_max + 1, y_min: y_max + 1]
        return cropped
    
def cal_metrics(labels: torch.Tensor, preds: torch.Tensor, wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    Compute metrics (loss, accuracy, MCC score, precision, recall, F1 score) using ground truth labels and logits.

    Arguments:
        labels (torch.Tensor): ground truth labels.
        preds (torch.Tensor): logits (not softmaxed yet).
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """
    
    # loss
    loss = F.cross_entropy(preds, labels)
    wandb_log[metric_prefix + 'loss'] = loss
        
    # get probability
    preds = torch.softmax(preds, axis = 1)

    # ROC AUC
    try:
        wandb_log[metric_prefix + 'auc'] = roc_auc_score(labels, preds, multi_class = 'ovr')
    except Exception:
        wandb_log[metric_prefix + 'auc'] = -1

    # get class prediction
    preds = preds.argmax(axis = 1)
    
    # accuracy and mcc
    for metric_name, metric_func in metrics_no_avg.items():
        metric = metric_func(labels, preds)
        wandb_log[metric_prefix + metric_name] = metric

    # precision, recall, f1 score
    for metric_name, metric_func in metrics_with_avg.items():
        metric = metric_func(labels, preds, average = avg, zero_division = 0)
        wandb_log[metric_prefix + metric_name] = metric


def plot_help(images: list[Image.Image | torch.Tensor], 
              titles: list[str], 
              figsize: tuple[int, int] = None, 
              grayscale: bool = True, 
              axis_off: bool = False
              ) -> None:
    assert(len(titles) == len(images))
    """
    Helper function for easy plotting in jupyter notebooks.

    Arguments:
        images (list[Image.Image | torch.Tensor]): images to plot.
        titles (list[str]): titles for images. 
        figsize (tuple[int, int]): figure size.
        grayscale (bool): whether to show image in RGB mode or grayscale mode.
        axis_off (bool): whether to disable axis.
    """
    
    # colormap
    cmap = 'gray' if grayscale else None
    
    # plot size
    if figsize is None:
        figsize = (len(titles) * 3 + 1, 3)

    f, axarr = plt.subplots(nrows = 1, ncols = len(titles), figsize = figsize)
    for a, t, i in zip(axarr, titles, images):
        a.set_title(t)
        if isinstance(i, Image.Image): # for PIL image
            a.imshow(i, cmap = cmap)
        elif isinstance(i, torch.Tensor): # for torch tensor
            i = i.detach().cpu()
            if len(i.shape) == 2: # torch tensor image of shape (h, w)
                a.imshow(i, cmap = cmap)
            elif len(i.shape) == 3: # torch tensor image of shape (1, h, w) or (3, h, w). Always channel-first.
                if i.shape[0] == 1:
                    a.imshow(i[0], cmap = cmap)
                elif i.shape[0] == 3:
                    a.imshow(i.permute(1, 2, 0))
                else:
                    raise Exception('wrong image dimension:', i.shape)
            else:
                raise Exception('wrong image dimension:', i.shape)
        else:
            raise Exception('wrong image type:', type(i))
    
        if axis_off:
            a.axis('off')
    
    plt.show()

def cal_IoUs(preds: torch.Tensor, targets: torch.Tensor, num_class: int = 4, eps: float = 1e-6) -> tuple[list, torch.Tensor]:
    """
    Calculate IoU per class and mean IoU. Reference: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy.

    Arguments:
        preds (torch.Tensor): predicted segmentation labels.
        targets (torch.Tensor): ground truth segmentation labels.
        num_class (int): number of unique classes in segmentation.
        eps (float): stabilizer.

    Returns:
        iou_per_class (list[torch.Tensor]): iou per class.
        miou (torch.Tensor): mean iou.
    """
 
    # preds and targets are of shape b * h * w
    iou_per_class = []
    
    for cls in range(num_class):
        pred_class = (preds == cls).float()
        true_class = (targets == cls).float()

        intersection = (pred_class * true_class).sum(dim=(1, 2))
        union = (pred_class + true_class).clamp(0, 1).sum(dim=(1, 2))

        iou = intersection / (union + eps)
        iou_per_class.append(iou)
    
    ious = torch.stack(iou_per_class, dim = 1)
    miou = ious.mean(dim = 1)

    return iou_per_class, miou

def area_opening(mask: torch.Tensor, area_threshold: int = 500, connectivity: int = 2) -> torch.Tensor:
    """
    Remove blobs in the mask.

    Arguments:
        mask (torch.Tensor): mask tensor.
        area_threshold (int): number of pixels in the removed area.
        connectivity (int): the maximum number of orthogonal steps to reach a neighbor.
    
    Returns:
        mask (torch.Tensor): area-opened mask.
    """

    device = mask.device
    mask = skimage.morphology.area_opening(mask.cpu().numpy()[0], area_threshold = area_threshold, connectivity = connectivity)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    return mask

import torch

def angular_distance(v1: torch.Tensor, v2: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Compute radian and degree distances between two normalized 3D vectors (i.e., gaze vectors).
    
    Arguments:
        v1 (torch.Tensor): tensor of shape (N, 3) - first set of 3D unit vectors, should be already normalized.
        v2 (torch.Tensor): tensor of shape (N, 3) - second set of 3D unit vectors, should be already normalized.
    
    Returns:
        radian (torch.Tensor): tensor of shape (N,) - radian distance.
        degree (torch.Tensor): tensor of shape (N,) - degree distance.
    """
    # compute dot product
    dot_product = torch.sum(v1 * v2, dim = 1)  # shape: (N,)

    # clamp to avoid numerical issues with acos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # compute angle in radians
    radian = torch.acos(dot_product)  # shape: (N,)
    
    # convert to degree
    degree = torch.rad2deg(radian)

    return radian, degree

def GramMatrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized gram matrix for feature map.

    Arguments:
        x (torch.Tensor): feature map.

    Returns:
        x (torch.Tensor): normalized gram matrix.
    """

    x = x.flatten(start_dim = -2) # flatten w and h of feature map
    n = x[0].numel() # number of elements in gram matrix
    x = x @ x.transpose(-2, -1) 
    x = x / n # normalize gram matrix
    return x

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