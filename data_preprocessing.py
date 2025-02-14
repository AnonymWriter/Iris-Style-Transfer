import os
import tqdm
import json
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.v2 as transforms

# self-defined functions
from utils import crop_image, cal_IoUs, area_opening
from models import RITnet, EfficientNet, ResNet50, extract_eye_landmarks
    
class OpenEDS2019IRDataset(torch.utils.data.Dataset):
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
                 area_threshold: int = 500, 
                 connectivity: int = 2,
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
            area_threshold (int): number of pixels in the removed area.
            connectivity (int): the maximum number of orthogonal steps to reach a neighbor.
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
            m = m_ritnet * m_glint
            # m = area_opening(m)
            x = x * m

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
    
class OpenEDS2019ISTDataset(torch.utils.data.Dataset):
    """
    Self-defined dataset, used for iris style transfer.
    """
    
    def __init__(self, 
                 c_imgs: list[torch.Tensor], 
                 c_labels: list[int],
                 c_masks_gt: list[torch.Tensor],
                 glint_threshold: float = 0.8,
                 area_threshold: int = 500, 
                 connectivity: int = 2,
                 ritnet: torch.nn.Module = None,
                 device: str = 'cuda:0',
                 ) -> None:
        """
        Arguments:
            c_imgs (list[torch.Tensor]): content image tensors. 
            c_labels (list[int]): content image class labels.
            c_masks_gt (list[torch.Tensor]): content image segmentation ground truth labels.
            glint_threshold (float): threshold for glints.
            area_threshold (int): number of pixels in the removed area.
            connectivity (int): the maximum number of orthogonal steps to reach a neighbor.
            ritnet (torch.nn.Module): RITnet model.
            device (str): CPU or GPU.
        """
        
        assert(len(c_imgs) == len(c_labels) == len(c_masks_gt))
        
        # regarding content images
        self.c_imgs = []
        self.c_labels = torch.as_tensor(c_labels).long().to(device)
        self.c_masks_gt = torch.stack(c_masks_gt).to(device)
        self.c_masks_iris = []
        self.c_iris_bbs = []
        self.ious0, self.ious1, self.ious2, self.ious3, self.mious = [], [], [], [], []
        
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
            
            # compute IoUs
            iou_per_class, miou = cal_IoUs(c_m_ritnet, c_m_gt.unsqueeze(0))
            self.ious0.append(iou_per_class[0])
            self.ious1.append(iou_per_class[1])
            self.ious2.append(iou_per_class[2])
            self.ious3.append(iou_per_class[3])
            self.mious.append(miou)

            # compute content image non-glint mask
            c_m_glint = c_img <= glint_threshold

            # apply both masks and compute bounding box
            c_m_ritnet = c_m_ritnet == 2
            c_m = c_m_ritnet * c_m_glint
            # c_m = area_opening(c_m)
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

def load_data_openeds2019(test_split_ratio: float = 0.2, 
                          load_seg: bool = False,
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
    Load OpenEDS2019 dataset.

    Arguments:
        test_split_ratio (float): train-test-split ratio.
        load_seg (bool): whether to load ground truth segmentation labels.
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
            
            # load eye image and get class label (each user is a class)
            p = i_folder + i_path
            img = Image.open(p).convert('L')
            img = t(img)
            img_class = img_class_dict[i_path] 
            img_train = img_train_dict[i_path] # whether this image is in training set or test set

            # load ground truth segmentation label
            if load_seg:
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

def load_data_openeds2020(extract_feature: bool,
                          estimator: int = 1, 
                          data_path: str = '../data/openeds2020/openEDS2020-GazePrediction/',
                          postfix: str = 'test/',
                          device: str = 'cpu',
                          ) -> tuple[
                                torch.Tensor, # images or segmentations or features
                                torch.Tensor, # landmarks
                                torch.Tensor, # gaze vectors
                                ]:
    """
    Load OpenEDS2020 dataset (gaze estimation part).

    Arguments:
        extract_feature (bool): whether to load features or images.
        estimator (int): 1 for model-based gaze estimator, 2 for appearance-based gaze estimator.
        data_path (str): dataset folder path.
        postfix (str): train, validation, or test.
        device (str): CPU or GPU.

    Returns:
        images (torch.Tensor): images or segmentations or features.
        labels (torch.Tensor): gaze vectors.
    """
    
    # transform
    t = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])

    # feature extractor
    if extract_feature:
        if estimator == 1: # model-based gaze estimator
            feature_extractor = EfficientNet()
        else: # estimator == 2, appearance-based gaze estimator
            feature_extractor = ResNet50()
        feature_extractor.to(device)
    
    images = []
    labels = []

    sequence_names = sorted(os.listdir(data_path + postfix + 'sequences/'))
    for sequence_name in sequence_names:
        # get sorted image names
        img_names = sorted(os.listdir(data_path + postfix + 'sequences/' + sequence_name))

        # read label
        label = pd.read_csv(data_path + postfix + 'labels/' + sequence_name + '.txt', header = None)
        label = label.iloc[:, 1:] # drop index column
        label = torch.tensor(label.values, dtype = torch.float32)

        # number of images should be equal to label length for train and valid sets, and 5 frames less for test set
        assert(len(img_names) == len(label) or len(img_names) == len(label) - 5)
        labels.append(label[:len(img_names)])
        
        for img_name in img_names:
            img = Image.open(data_path + postfix + 'sequences/' + sequence_name + '/' + img_name).convert('L')
            img = t(img)
            
            if extract_feature:
                img = img.to(device)
                img = feature_extractor(img)
                img = img[0].cpu()
                
                if estimator == 1: # further extract landmarks from segmentations for model-based gaze estimator
                    img = extract_eye_landmarks(img)
                
            images.append(img)

    images = torch.stack(images)
    labels = torch.cat(labels)
    
    return images, labels