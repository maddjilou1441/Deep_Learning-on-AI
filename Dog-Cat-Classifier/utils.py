from __future__ import print_function

import argparse
import csv
import os
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy
from random import shuffle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

import glob
import cv2

# The module also provides a number of factory functions, 
# including functions to load images from files, and to create new images.
from PIL import Image