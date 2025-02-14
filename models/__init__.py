from .vgg.vgg import VGG19
from .ritnet.ritnet import RITnet
from .resnet.resnet import ResNet50
from .efficientnet.efficientnet import EfficientNet
from .classifiers.classifiers import Classifier1, Classifier2
from .gaze_estimators.gaze_estimators import GazeEstimator1, GazeEstimator2, extract_eye_landmarks, GazeEstimator1_complicated

name = 'models'