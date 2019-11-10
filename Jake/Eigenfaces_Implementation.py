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

def getFaceDeviation(face_images, average_face):
    face_deviations = []
    for face in face_images:
        face_deviations.append(face.image_array - average_face)
    return face_deviations

def getEigenVectors(face_deviations):
    A = np.concatenate(face_deviations, axis=1)
    L = np.dot(np.transpose(A),A)
    return np.linalg.eig(L)

def main():
    face_images = importDataSet(os.getcwd() + '/Jake/faces_dataset')
    print(len(face_images))
    average_face = getAverageFace(face_images)
    face_deviations = getFaceDeviation(face_images, average_face)
    eigen_values, eigen_vectors = getEigenVectors(face_deviations)
    print(eigen_values)
    print(len(eigen_values))
    print(eigen_vectors)
    print(len(eigen_vectors))

main()