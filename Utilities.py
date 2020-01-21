# Utility methods used in each recognition implementation

from skimage import io
from skimage import transform
from face_image import FaceImage
from eigen_pair import EigenPair
import numpy as np
import os
import math


def importDataSet(foldername=os.getcwd() + "/Face_Images/faces_dataset"):
    face_images = []
    for filename in os.listdir(foldername):
        path = foldername + '/' + filename
        face_images.append(FaceImage(path, filename))
    return face_images


def euclideanDistance(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        return None
    distance = 0.0
    for i in range(len(vector_1) - 1):
        distance += (vector_1[i] - vector_2[i])**2
    return math.sqrt(distance)
