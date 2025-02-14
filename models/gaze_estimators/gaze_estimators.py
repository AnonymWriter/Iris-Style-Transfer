import cv2
import torch
import numpy as np

# self-defined functions
from models import ResNet50

class GazeEstimator1(torch.nn.Module):
    """
    A simple model-based gaze estimator.
    """

    def __init__(self, extract_feature: bool = False, landmark_dim: int = 19, hidden_dim: int = 64, output_dim: int = 3):
        """
        Arguments:
            extract_feature (bool): whether to extract eye landmark features.
            landmark_dim (int): dimension of landmark features.
            hidden_dim (int): dimension of hidden layer.
            output_dim (int): dimension of output (3D gaze vector).
        """

        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features = landmark_dim, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = output_dim, bias = True)
        )
        
        self.extract_feature = extract_feature
    
    def forward(self, x) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): segmentation maps of shape (b, 400, 640) or (b, 1, 400, 640), or landmark features of shape (b, 19).

        Returns:
            x (torch.Tensor): normalized 3D gaze vectors.
        """

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        if self.extract_feature:
            x = torch.stack([extract_eye_landmarks(i[0]) for i in x]) # extract_eye_landmarks can only process one image at a time    
            
        x = self.model(x)
        x = x / torch.norm(x, dim = 1, keepdim = True) # normalize
        return x

def find_ellipse_features(mask: np.array) -> tuple[float]:
    """ 
    Fits an ellipse and returns center, major/minor axes, and angle.
    
    Arguments:
        mask (np.array): binary mask, either for pupil or iris.
        
    Returns:
        cx (float): ellipse center x.
        cy (float): ellipse center y.
        major_axis (float): ellipse major axis.
        minor_axis (float): ellipse minor axis.
        angle (float): ellipse angle. 
    """
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None, None, None  # no valid contour
    max_contour = max(contours, key = cv2.contourArea)
    
    # ensure at least 5 points to fit an ellipse
    if len(max_contour) < 5:
        return None, None, None, None, None
    
    # fit ellipse
    ellipse = cv2.fitEllipse(max_contour)
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    return cx, cy, major_axis, minor_axis, angle
    
def find_eye_corners(mask: np.array) -> tuple[int]:
    """
    Find eye corners based on sclera mask.
    
    Arguments:
        mask (np.array): binary mask for sclera.
        
    Returns:
        left_corner (int): left corner of eye.
        right_corner (int): right corner of eye.
        bottom_corner (int): bottom corner of eye.
        top_corner (int): top corner of eye.
    """
    
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None, None, None, None # no eye region detected
    left_corner = min(x_indices)
    right_corner = max(x_indices)
    bottom_corner = min(y_indices)
    top_corner = max(y_indices)
    return left_corner, right_corner, bottom_corner, top_corner
        
def extract_eye_landmarks(segmentation: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Extracts eye landmarks.
    
    Arguments:
        segmentation (torch.Tensor): output of EfficientNet. 
        epsilon (float): small value to avoid division by zero.
    
    Returns:
        landmarks (torch.Tensor): eye landmark features.
    """
    
    # segmentation has to be of shape (400, 640)
    assert(segmentation.shape == (400, 640))
    
    # device
    device = segmentation.device
    
    # extract binary masks
    segmentation = segmentation.cpu().numpy().astype(np.uint8)
    sclera_mask = (segmentation == 1).astype(np.uint8)
    iris_mask = (segmentation == 2).astype(np.uint8)
    pupil_mask = (segmentation == 3).astype(np.uint8)

    # get pupil and iris features
    pupil_center_x, pupil_center_y, pupil_major, pupil_minor, pupil_angle = find_ellipse_features(pupil_mask)
    iris_center_x, iris_center_y, iris_major, iris_minor, iris_angle = find_ellipse_features(iris_mask)

    # eye corners
    left_corner, right_corner, bottom_corner, top_corner = find_eye_corners(sclera_mask)

    # eye width, height, and eye aspect ratio
    if left_corner is not None:
        eye_width = right_corner - left_corner
        eye_height = top_corner - bottom_corner
        ear = eye_height / (eye_width + epsilon)
    else:
        eye_width, eye_height, ear = None, None, None

    # normalized pupil position
    if pupil_center_x is not None and left_corner is not None:
        norm_pupil_x = (pupil_center_x - (left_corner + right_corner) / 2) / (eye_width + epsilon)
        norm_pupil_y = (pupil_center_y - (bottom_corner + top_corner) / 2) / (eye_height + epsilon)
    else:
        norm_pupil_x, norm_pupil_y = None, None

    landmarks = [
        pupil_center_x,     # pupil center x
        pupil_center_y,     # pupil center y
        pupil_major,        # pupil major axis
        pupil_minor,        # pupil minor axis
        pupil_angle,        # pupil angle
        iris_center_x,      # iris center x
        iris_center_y,      # iris center y
        iris_major,         # iris major axis
        iris_minor,         # iris minor axis
        iris_angle,         # iris angle
        left_corner,        # left eye corner
        right_corner,       # right eye corner
        bottom_corner,      # bottom eye corner
        top_corner,         # top eye corner
        eye_width,          # eye width
        eye_height,         # eye height
        ear,                # eye aspect ratio
        norm_pupil_x,       # normalized pupil x
        norm_pupil_y,       # normalized pupil y
    ]
    
    landmarks = [0 if l is None else l for l in landmarks]
    landmarks = torch.tensor(landmarks, dtype = torch.float32).to(device)
    return landmarks

class GazeEstimator2(torch.nn.Module):
    """
    A simple appearancemodel-based gaze estimator.
    """

    def __init__(self, extract_feature: bool = False, freeze_resnet: bool = True, hidden_dim: int = 64, output_dim: int = 3):
        """
        Arguments:
            extract_feature (bool): whether to load the ResNet50 model and use it to extract features.
            freeze_resnet (bool): whether to freeze the ResNet50 model.
            hidden_dim (int): dimension of hidden layer.
            output_dim (int): dimension of output (3D gaze vector).
        """

        super().__init__()
              
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features = 2048, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = output_dim, bias = True)
            )
        
        self.extract_feature = extract_feature
        if self.extract_feature:
            self.resnet = ResNet50(freeze = freeze_resnet)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x (torch.Tensor): image tensor.

        Returns:
            x (torch.Tensor): normalized 3D gaze vectors.
        """

        if self.extract_feature:
            x = self.resnet(x) 
        x = self.model(x)
        x = x / torch.norm(x, dim = 1, keepdim = True) # normalize
        return x
    
class GazeEstimator1_complicated(torch.nn.Module):
    """
    A simple model-based gaze estimator.
    """

    def __init__(self, extract_feature: bool = False, landmark_dim: int = 19, hidden_dim: int = 64, output_dim: int = 3):
        """
        Arguments:
            extract_feature (bool): whether to extract eye landmark features.
            landmark_dim (int): dimension of landmark features.
            hidden_dim (int): dimension of hidden layer.
            output_dim (int): dimension of output (3D gaze vector).
        """

        super().__init__()
        
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_dim, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 1, stride = 1, padding = 1),
            torch.nn.BatchNorm2d(hidden_dim),
            torch.nn.ReLU(inplace = True),
            torch.nn.AdaptiveAvgPool2d(output_size = (1, 1)),

            torch.nn.Flatten()
        )
        
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(in_features = landmark_dim, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = True),
        )
        
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(in_features = hidden_dim + hidden_dim, out_features = hidden_dim, bias = True),
            torch.nn.ReLU(inplace = True),
            torch.nn.Dropout(p = 0.5, inplace = False),
            torch.nn.Linear(in_features = hidden_dim, out_features = output_dim, bias = True)
        )
        
        self.extract_feature = extract_feature
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            x1 (torch.Tensor): segmentation maps of shape (b, 400, 640) or (b, 1, 400, 640).
            x2 (torch.Tensor): landmark features of shape (b, 19).

        Returns:
            x (torch.Tensor): normalized 3D gaze vectors.
        """

        assert(len(x1.shape) >= 3)
        if len(x1.shape) == 3:
            x1 = x1.unsqueeze(1)
            
        if self.extract_feature:
            x2 = torch.stack([extract_eye_landmarks(i[0]) for i in x1]) # extract_eye_landmarks can only process one image at a time    
            
        # segmentation map is tensor of int/long, convert to float
        x1 = x1.float()    
        
        x1 = self.model1(x1)            
        x2 = self.model2(x2)
        x = torch.cat([x1, x2], dim = 1)
        x = self.projection(x)
        x = x / torch.norm(x, dim = 1, keepdim = True) # normalize
        return x