# Utility methods used in each recognition implementation

from skimage import io
from skimage import transform
import numpy as np
import os
import math

class FaceImage:
        image_array = None
        image_vector = None
        identity = None
        OMEGA_k = None

        def __init__(self, image, identity):
            if isinstance(image, str):
                self.image_array = np.array(transform.resize(io.imread(image, as_gray=True), (150,150)))
            else:
                self.image_array = image
            if isinstance(identity, str):
                identity = identity.split('_')[0]
                if identity == "unidentified":
                    self.identity = None
                else:
                    self.identity = int(identity)
                
            else:
                self.identity = identity
            self.image_vector = self.image_array.flatten().reshape(-1,1)
            self.OMEGA_k = None
        
        def displayImage(self):
            io.imshow(self.image_array)

class EigenPair:
    eigen_value = None
    magnitude = None
    eigen_vector = None

    def __init__(self, eigen_value, eigen_vector):
        self.eigen_value = eigen_value
        self.magnitude = abs(eigen_value)
        self.eigen_vector = eigen_vector

@staticmethod
def importDataSet(foldername = os.getcwd() + "/../Face_Images/faces_dataset"):
    face_images = []
    for filename in os.listdir(foldername):
        path = foldername + '/' + filename
        face_images.append(FaceImage(path, filename))
    return face_images

@staticmethod
def euclideanDistance(vector1, vector2):
    if len(vector1) != len(vector2):
        return None
    distance = 0.0
    for i in range(len(vector1) - 1):
        distance += (vector1[i] - vector2[i])**2
    return math.sqrt(distance)