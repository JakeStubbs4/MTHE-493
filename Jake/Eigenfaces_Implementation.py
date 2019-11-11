# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import os

class FaceImage:
    image_array = None
    identity = None

    def __init__(self, image, identity):
        if isinstance(image, str):
            self.image_array = np.array(io.imread(image, as_gray=True))
        else:
            self.image_array = image
        self.identity = identity

class EigenPair:
    eigen_value = None
    magnitude = None
    eigen_vector = None

    def __init__(self, eigen_value, eigen_vector):
        self.eigen_value = eigen_value
        self.magnitude = abs(eigen_value)
        self.eigen_vector = eigen_vector

def importDataSet(foldername):
    face_images = []
    identity = 0
    for filename in os.listdir(foldername):
        path = foldername + '/' + filename
        face_images.append(FaceImage(path, identity))
        identity = identity + 1
    return face_images

def getAverageFace(face_images):
    face_images_arrays = []
    for image in face_images:
        face_images_arrays.append(image.image_array)
    return np.mean(face_images_arrays, axis=0)

def getFaceDeviations(face_images, average_face):
    face_deviations = []
    for face in face_images:
        face_deviations.append(face.image_array - average_face)
    return face_deviations

def covarianceEigenvectors(face_deviations):
    A = np.concatenate(face_deviations, axis=1)
    L = np.dot(np.transpose(A),A)
    return np.linalg.eig(L)

def main():
    face_images = importDataSet(os.getcwd() + '/Jake/faces_dataset')
    print(len(face_images))
    average_face = getAverageFace(face_images)
    face_deviations = getFaceDeviations(face_images, average_face)
    eigen_values, eigen_vectors = covarianceEigenvectors(face_deviations)
    print("eigenvalues: ")
    print(eigen_values)
    print("eigenvectors: ")
    print(eigen_vectors)
    '''eigen_pairs = []
    for i in range(len(eigen_values)):
        eigen_pairs.append(EigenPair(eigen_values[i], eigen_vectors[i]))
    eigen_pairs.sort(key=lambda x: x.magnitude, reverse=True)
    for k in range(20):
        print(eigen_pairs[k].magnitude)'''


main()