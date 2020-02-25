# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from matplotlib import pyplot as plt
from random import randrange
import numpy as np
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

# Converts eigen vector to face images to be displayed.
def getEigenFace(eigen_vector, K):
    eigen_face = np.dot(K, eigen_vector).reshape(150, 150)
    return eigen_face

# Projects newly introduced face image onto predetermined low dimensional image space.
def projectImage(face_image, eigen_pairs, K):
    projection = []
    for pair in eigen_pairs:
        omega_k = float(np.dot(np.dot(K, pair.eigen_vector), face_image))
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

# Apply the kernel function to two given data points Xi and Xj with parameter d.
def applyKernel(Xi, Xj, d):
    result = float((np.dot(np.transpose(Xi), Xj))**d)
    return result

# Construct an NxN Kernel matrix by applying the kernel function to each of the N data points with respect to each data point.
def kernelMatrix(face_images, d):
    K = []
    for i in range(len(face_images)):
        K.append([])
        for j in range(len(face_images)):
            K[i].append(applyKernel(face_images[i].image_vector, face_images[j].image_vector, d))
    return K

# Normalize the Kernel matrix using the Gram matrix approach
def normalizeKernel(K):
    N = len(K[0])
    N_1 = np.full((N,N), 1/N)
    Khat = K - np.matmul(N_1, K) - np.matmul(K, N_1) + np.matmul(np.matmul(N_1, K), N_1)
    return Khat

# Recover the kernel principle components from the resulting eigenvectors of the kernel matrix
def projectToKernelSpace(current_face_image, eigen_vectors, face_images, d):
    projection = eigen_vectors[0].eigen_vector*applyKernel(current_face_image, face_images[0].image_vector, d)
    for i in range(1, len(eigen_vectors)):
        projection += eigen_vectors[i].eigen_vector*applyKernel(current_face_image, face_images[i].image_vector, d)
    return projection

def getError(face_images, kernel_parameters, eigenspace_dimension):
    # Calculate K matrix
    K = kernelMatrix(face_images, kernel_parameters)

    # Normalize Kernel Matrix
    K = normalizeKernel(K)

    # Calculate eigen vectors and values from the Normalized Kernel Matrix.
    eigen_values, eigen_vectors = np.linalg.eig(K)
    print(eigen_values)

    # Pair the eigenvectors and eigenvalues then order pairs by decreasing eigenvalue magnitude.
    eigen_pairs = []
    for i in range(len(eigen_values)):
        eigen_pairs.append(EigenPair(eigen_values[i], eigen_vectors[i]))
    eigen_pairs.sort(key=lambda x: x.magnitude, reverse=True)

    # Measure the resulting error from neglecting the remaining (len(eigen_pairs) - dim) eigen vectors.
    error = 0
    for k in range(eigenspace_dimension + 1, len(eigen_pairs) - 1):
        error += eigen_pairs[k].magnitude
    return error

def optimize_kernel(face_images, eigenspace_dimension):
    kernel_dimension = 0
    print(f"INITIAL KERNEL DIMENSION: {kernel_dimension}")
    delta = 0.00000000000001           # WHAT TO MAKE DELTA?
    learning_rate = 0.001    # WHAT TO MAKE LEARNING RATE?
    precision = 1
    delta_dimension = precision + 1
    max_iterations = 1000
    iterations = 0
    current_cost = getError(face_images, kernel_dimension, eigenspace_dimension)
    while delta_dimension > precision and iterations < max_iterations:
        previous_cost = current_cost
        current_cost = getError(face_images, kernel_dimension + delta, eigenspace_dimension)
        cost_derivative = (current_cost - previous_cost)/delta
        previous_dimension = kernel_dimension
        kernel_dimension = kernel_dimension - learning_rate*cost_derivative
        delta_dimension = abs(previous_dimension - kernel_dimension)
        iterations += 1
        print(f"At Iteration: {iterations}, Kernel Dimension is: {kernel_dimension}")
    return kernel_dimension

def identify(face_images, kernel_dimension, eigenspace_dimension, num_nearest_neighbors):
    # Calculate K matrix
    K = kernelMatrix(face_images, kernel_dimension)

    # Normalize Kernel Matrix
    K = normalizeKernel(K)

    # Calculate eigen vectors and values from the Normalized Kernel Matrix.
    eigen_values, eigen_vectors = np.linalg.eig(K)
    print(eigen_values)

    # Pair the eigenvectors and eigenvalues then order pairs by decreasing eigenvalue magnitude.
    eigen_pairs = []
    for i in range(len(eigen_values)):
        eigen_pairs.append(EigenPair(eigen_values[i], eigen_vectors[i]))
    eigen_pairs.sort(key=lambda x: x.magnitude, reverse=True)

    # Choose a subset of eigenpairs corresponding to OPTIMAL_DIM largest eigenvalues.
    ms_eigen_pairs = []
    for k in range(eigenspace_dimension):
        ms_eigen_pairs.append(eigen_pairs[k])

    # Classify the given training dataset based on the chosen subspace.
    for face in face_images:
        face.OMEGA_k = projectToKernelSpace(face.image_vector, ms_eigen_pairs, face_images, kernel_dimension)

    # Introduce new face and classify
    new_face_file = input("Enter the filename of an image to be classified: ")
    new_face = FaceImage(new_face_file, None)

    new_face_projection = projectToKernelSpace(new_face.image_vector, ms_eigen_pairs, face_images, kernel_dimension)

    corresponding_faces = KNearestNeighbors(face_images, new_face_projection, num_nearest_neighbors)
    for face in corresponding_faces:
        print(face.identity)

    corresponding_face = classifyImage(corresponding_faces, new_face_projection)

    plt.figure(1)
    plt.title("Unidentified")
    new_face.displayImage()

    plt.figure(2)
    plt.title("Possible Match")
    corresponding_face.displayImage()

    plt.show()

def main():
    # Optimal Eigenspace dimension
    OPTIMAL_DIMENSION = 7
    # Optimal nearest neighbors to consider.
    OPTIMAL_NEAREST_NEIGHBORS = 3
    # Import training data set.
    face_images = importDataSet()
    
    KERNEL_PARAMETERS = optimize_kernel(face_images, OPTIMAL_DIMENSION)
    identify(face_images, KERNEL_PARAMETERS, OPTIMAL_DIMENSION, OPTIMAL_NEAREST_NEIGHBORS)


if __name__ == "__main__":
    main()