import os
import json
import torch
import random
import shutil
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
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

def read_data(test_split_ratio: float = 0.2, 
              read_seg: bool = False,
              image_paths: list[str] = ['../data/openeds2019/Semantic_Segmentation_Dataset/train/images/', 
                                        '../data/openeds2019/Semantic_Segmentation_Dataset/validation/images/',
                                        '../data/openeds2019/Semantic_Segmentation_Dataset/test/images/'],
              json_paths:  list[str] = ['../data/openeds2019/OpenEDS_train_userID_mapping_to_images.json', 
                                        '../data/openeds2019/OpenEDS_validation_userID_mapping_to_images.json',
                                        '../data/openeds2019/OpenEDS_test_userID_mapping_to_images.json'],
              seg_paths:   list[str] = ['../data/openeds2019/Semantic_Segmentation_Dataset/train/labels/', 
                                        '../data/openeds2019/Semantic_Segmentation_Dataset/validation/labels/',
                                        '../data/openeds2019/Semantic_Segmentation_Dataset/test/labels/'],
            ) -> tuple[
                    list[torch.Tensor], # train images tensors 
                    list[int],          # train image class labels
                    list[torch.Tensor], # train ground truth segmentation labels
                    list[torch.Tensor], # test images tensors
                    list[int],          # test image class labels
                    list[torch.Tensor], # test ground truth segmentation labels
                    int                 # number of classes
                ]:
    """
    Read OpenEDS2019 dataset.

    Arguments:
        test_split_ratio (float): train-test-split ratio.
        read_seg (bool): whether to read ground truth segmentation labels.
        image_paths (list[str]): image folder paths.
        json_paths (list[str]): user-image mapping json file paths.
        seg_paths (list[str]): grount truth segmentation folder paths.

    Returns:
        train_x (list[torch.Tensor]): train images tensors. 
        train_y (list[int]): train image class labels.
        train_m (list[torch.Tensor]): train ground truth segmentation labels.
        test_x (list[torch.Tensor]): test images tensors.
        test_y (list[int]): test image class labels.
        test_m (list[torch.Tensor]): test ground truth segmentation labels
        class_count (int): number of classes.
    """
    
    train_x, train_y, train_m, test_x, test_y, test_m = [], [], [], [], [], []
    class_count = 0
    
    # PIL to tensor
    t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])

    for i_folder, j_path, m_folder in zip(image_paths, json_paths, seg_paths):
        with open(j_path, 'r') as file:
            mappings= json.load(file)
        
        # create image-class and image-split dictionaries
        img_class_dict = {}
        img_train_dict = {}
        for m in mappings:
            # id = m['id']
            imgs = m['semantic_segmenation_images']
            if len(imgs) <= 2: # skip users with too few samples
                continue
            
            train_imgs, test_imgs = torch.utils.data.random_split(imgs, [1 - test_split_ratio, test_split_ratio])
            for i in range(len(imgs)):
                img_class_dict[imgs[i]] = class_count
                img_train_dict[imgs[i]] = i in train_imgs.indices
            class_count += 1

        # load images and determine their classes
        img_paths = os.listdir(i_folder)
        for i_path in img_paths:
            if i_path not in img_class_dict: # skipped users
                continue
            
            # read eye image and get class label (each user is a class)
            p = i_folder + i_path
            img = Image.open(p).convert('L')
            img = t(img)
            img_class = img_class_dict[i_path] 
            img_train = img_train_dict[i_path] # whether this image is in training set or test set

            # read ground truth segmentation label
            if read_seg:
                m_path = i_path[:-4] + '.npy' # file name from .jpg to .npy
                img_mask = torch.from_numpy(np.load(m_folder + m_path))
            else:
                img_mask = None
            
            if img_train:
                train_x.append(img)
                train_y.append(img_class)
                train_m.append(img_mask)
            else:
                test_x.append(img)
                test_y.append(img_class)
                test_m.append(img_mask)
    
    return train_x, train_y, train_m, test_x, test_y, test_m, class_count

def cal_mIoU(preds: torch.Tensor, targets: torch.Tensor, num_class: int = 4) -> float:
    """
    Calculate mean IoU over classes. Reference: https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy.

    Arguments:
        preds (torch.Tensor): predicted segmentation labels.
        targets (torch.Tensor): ground truth segmentation labels.
        num_class (int): number of unique classes in segmentation.
    """
 
    # n * h * w
    iou_per_class = []
    
    for cls in range(num_class):
        pred_class = (preds == cls).float()
        true_class = (targets == cls).float()

        intersection = (pred_class * true_class).sum(dim=(1, 2))
        union = (pred_class + true_class).clamp(0, 1).sum(dim=(1, 2))

        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou)
    
    iou_per_class = torch.stack(iou_per_class, dim = 1)
    mean_iou = iou_per_class.mean(dim = 1)

    return mean_iou

def sample_other(label: int, labels: list[int]) -> int:
    """
    Given a class label, sample a random sample of another class.

    Arguments:
        label (int): class label.
        labels (list[int]): sample label list.

    Returns:
        idx (int): index of sample.
    """
    idx = random.randrange(len(labels))
    while labels[idx] == label:
        idx = random.randrange(len(labels))
    return idx