# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from matplotlib import pyplot as plt
import numpy as np
import os
from utilities import euclideanDistance, importDataSet, FaceImage, EigenPair, KNearestNeighbors

# Computes the vector representation of the average face of all of the faces in the provided dataset.


def averageVector(face_images):
    face_images_arrays = []
    for image in face_images:
        face_images_arrays.append(image.image_array)
    return np.mean(face_images_arrays, axis=0).reshape(-1, 1)

# Computes the standard deviation of each face image and returns an array of deviation vectors.


def standardDeviation(face_images, average_face):
    face_deviations = []
    for face in face_images:
        face_deviations.append(face.image_vector - average_face)
    return face_deviations

# Computes the eigenvectors of the provided empirical covariance matrix A.


def covarianceEigenvectors(face_deviations, A):
    L = np.dot(np.transpose(A), A)
    return np.linalg.eig(L)

# Converts eigen vector to face images to be displayed.


def getEigenFace(eigen_vector, A):
    eigen_face = np.dot(A, eigen_vector).reshape(150, 150)
    return eigen_face

# Projects newly introduced face image onto predetermined low dimensional image space.


def projectImage(face_image, eigen_pairs, average_face, A):
    projection = []
    for pair in eigen_pairs:
        omega_k = float(np.dot(np.dot(A, pair.eigen_vector), face_image - average_face))
        projection.append(omega_k)
    return projection

# Classify unidentified face image projection based on the projections of the identified K nearest neighbors.


def classifyImage(corresponding_faces, new_face_projection):
    identity_dictionary = dict()
    for faceImage in corresponding_faces:
        if identity_dictionary.get(faceImage.identity) == None:
            identity_dictionary[faceImage.identity] = faceImage
        else:
            identity_dictionary[faceImage.identity].OMEGA_k = np.mean([identity_dictionary[faceImage.identity].OMEGA_k, faceImage.OMEGA_k], axis=0)

    updated_results = []
    for key in identity_dictionary:
        dist = euclideanDistance(new_face_projection, identity_dictionary[key].OMEGA_k)
        updated_results.append((identity_dictionary[key], dist))

    updated_results.sort(key=lambda tup: tup[1])
    return updated_results[0][0]

def identify(face_images, ms_eigen_pairs, OPTIMAL_K, average_face, A, unidentified_image=None):

    if (unidentified_image == None):
        # Introduce new face and classify
        new_face_file = input("Enter the filename of an image to be classified: ")
        new_face = FaceImage(new_face_file, None)
    else:
        new_face = unidentified_image

    new_face_projection = projectImage(new_face.image_vector, ms_eigen_pairs, average_face, A)

    corresponding_faces = KNearestNeighbors(face_images, new_face_projection, OPTIMAL_K)
    for face in corresponding_faces:
        print(face.identity)

    corresponding_face = classifyImage(corresponding_faces, new_face_projection)

    if (unidentified_image == None):
        plt.figure(2)
        plt.title("Unidentified")
        new_face.displayImage()

        plt.figure(3)
        plt.title("Possible Match")
        corresponding_face.displayImage()

        plt.show()

    else:
        print(f"Corresponding Face: {corresponding_face.identity}")
        print(f"Unidentified Face: {new_face.identity}")
        if (corresponding_face.identity == new_face.identity):
            return 1
        else:
            return 0


def main():
    '''IMPORT DATA SET AND TRAIN'''
    # Import training data set.
    face_images = importDataSet()

    # Compute the average of all of the imported face images.
    average_face = averageVector(face_images)

    # Compute the deviation of all of the face images.
    face_deviations = standardDeviation(face_images, average_face)

    # Calculate A matrix, impirical covariance matrix is given by C = A*AT
    A = np.concatenate(face_deviations, axis=1)

    # Calculate eigen vectors and values from the impirical covariance matrix.
    eigen_values, eigen_vectors = covarianceEigenvectors(face_deviations, A)

    # Pair the eigenvectors and eigenvalues then order pairs by decreasing eigenvalue magnitude.
    eigen_pairs = []
    for i in range(len(eigen_values)):
        eigen_pairs.append(EigenPair(eigen_values[i], eigen_vectors[i]))
    eigen_pairs.sort(key=lambda x: x.magnitude, reverse=True)

    # Optimal dimension for accuracy of recognition.
    OPTIMAL_DIM = 7
    # Optimal nearest neighbors to consider for accuracy of recognition.
    OPTIMAL_K = 3
    # Choose a subset of eigenpairs corresponding to DIM largest eigenvalues.

    ms_eigen_pairs = []
    for k in range(OPTIMAL_DIM):
        ms_eigen_pairs.append(eigen_pairs[k])

    error = 0
    for k in range(OPTIMAL_DIM + 1, len(eigen_pairs) - 1):
        error += eigen_pairs[k].magnitude
    print(f"Residual error of eigenfaces is: {error}")

    # Classify the given training dataset based on the chosen subspace.
    for face in face_images:
        face.OMEGA_k = projectImage(face.image_vector, ms_eigen_pairs, average_face, A)
        print(face.OMEGA_k)

    unidentified_images = importDataSet(os.getcwd() + "/Face_Images/unidentified", True)
    performance_vector = []
    for unidentified_image in unidentified_images:
        performance_vector.append(identify(face_images, ms_eigen_pairs, OPTIMAL_K, average_face, A, unidentified_image))
    print(f"The resulting algorithm achieves {(sum(performance_vector)/len(performance_vector))*100}% recognition accuracy.")


    # TODO: Add some check which will determine if the match is close enough.
    #new_face.identity = corresponding_face[0].identity


if __name__ == "__main__":
    main()