# Utility methods used in each recognition implementation

from skimage import io
from skimage import transform
from face_image import FaceImage
from eigen_pair import EigenPair
import numpy as np
import os
import math


def importDataSet(foldername=os.getcwd() + "/Face_Images/faces_dataset", unidentified_flag=False):
    face_images = []
    for filename in os.listdir(foldername):
        path = foldername + '/' + filename
        if unidentified_flag:
            face_images.append(FaceImage(path, int(filename.split('_')[1].split('.')[0])))
        else:
            face_images.append(FaceImage(path, filename))
    return face_images


def euclideanDistance(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        return None
    distance = 0.0
    for i in range(len(vector_1) - 1):
        distance += (vector_1[i] - vector_2[i])**2
    return math.sqrt(abs(distance))

# K Nearest Neighbor to classify an individual image projection based on the shortest euclidean distance from the projected training images.


def KNearestNeighbors(training_classes, test_row, num_neighbors):
    distances = list()
    for face in training_classes:
        dist = euclideanDistance(test_row, face.OMEGA_k)
        distances.append((face, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
