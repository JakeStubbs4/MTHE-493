# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from matplotlib import pyplot as plt
from random import randrange
import numpy as np
import math
import os
from utilities import euclideanDistance, importDataSet, FaceImage, EigenPair, KNearestNeighbors

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

# Apply a linear combination of the Polynomial kernel with parameters d=dimension and c=offset and the gaussian kernel with parameter Sigma to data points Xi and Xj.
def applyKernel(Xi, Xj, alpha, d, c, beta, Sigma):
    result = alpha*float((np.dot(np.transpose(Xi), Xj) + c)**d) + beta*float(math.exp(-1*(np.linalg.norm(Xi - Xj)**2)/(2*(Sigma**2))))
    return result

# Construct an NxN Kernel matrix by applying the kernel function to each of the N data points with respect to each data point.
def kernelMatrix(face_images, alpha, d, c, beta, Sigma):
    K = []
    for i in range(len(face_images)):
        K.append([])
        for j in range(len(face_images)):
            K[i].append(applyKernel(face_images[i].image_vector, face_images[j].image_vector, alpha, d, c, beta, Sigma))
    return K

# Normalize the Kernel matrix using the Gram matrix approach
def normalizeKernel(K):
    N = len(K[0])
    N_1 = np.full((N,N), 1/N)
    Khat = K - np.matmul(N_1, K) - np.matmul(K, N_1) + np.matmul(np.matmul(N_1, K), N_1)
    return Khat

# Recover the kernel principle components from the resulting eigenvectors of the kernel matrix
def getPrincipleComponent(current_face_image, eigen_vector, face_images, alpha, d, c, beta, Sigma):
    principle_component = eigen_vector.eigen_vector[0]*applyKernel(current_face_image, face_images[0].image_vector, alpha, d, c, beta, Sigma)
    for i in range(1, len(face_images)):
        principle_component += eigen_vector.eigen_vector[i]*applyKernel(current_face_image, face_images[i].image_vector, alpha, d, c, beta, Sigma)
    return principle_component

# Project an image to the kernal space.
def projectToKernelSpace(current_face_image, eigen_vectors, face_images, alpha, d, c, beta, Sigma):
    projection = []
    for i in range(0, len(eigen_vectors)):
        projection.append(getPrincipleComponent(current_face_image, eigen_vectors[i], face_images, alpha, d, c, beta, Sigma))
    return projection


def getError(face_images, kernel_parameters, eigenspace_dimension):
    # Calculate K matrix
    K = kernelMatrix(face_images, kernel_parameters[0], kernel_parameters[1], kernel_parameters[2], kernel_parameters[3], kernel_parameters[4])

    # Normalize Kernel Matrix
    K = normalizeKernel(K)

    # Calculate eigen vectors and values from the Normalized Kernel Matrix.
    eigen_values, eigen_vectors = np.linalg.eig(K)

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
    kernel_alpha = 1
    kernel_dimension = 1
    kernel_offset = 0
    kernel_beta = 1
    kernel_Sigma = 1
    kernel_parameters = [kernel_alpha, kernel_dimension, kernel_offset, kernel_beta, kernel_Sigma]
    min_kernel_parameters = kernel_parameters
    min_error = getError(face_images, kernel_parameters, eigenspace_dimension)
    print(f"INITIAL KERNEL PARAMETERS: {kernel_parameters}")
    delta = 0.00000001
    precision = 0.0001
    accuracy = precision + 1
    max_iterations = 25
    residual_errors = []
    iterations = 1
    da = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [delta, 0, 0, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    dd = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, delta, 0, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    dc = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, delta, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    db = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, 0, delta, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    ds = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, 0, 0, delta])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    while accuracy > precision and iterations <= max_iterations:
        prev_parameters = kernel_parameters
        cost_vector = [da(prev_parameters), dd(prev_parameters), dc(prev_parameters), db(prev_parameters), ds(prev_parameters)]
        learning_rate_vector = np.multiply(1/(iterations), [1/(abs(cost_vector[0]) + delta), 1/(abs(cost_vector[1]) + delta), 1/(abs(cost_vector[2]) + delta), 1/(abs(cost_vector[3]) + delta), 1/(abs(cost_vector[4]) + delta)])
        # learning_rate_vector = np.multiply(1/iterations, cost_vector)
        error = getError(face_images, kernel_parameters, eigenspace_dimension)
        residual_errors.append(error)
        print(f"At iteration {iterations} the kernel parameters are: {kernel_parameters}, residual error is: {error}, accuracy is {accuracy}")
        kernel_parameters = prev_parameters - np.multiply(learning_rate_vector, cost_vector)
        accuracy = np.linalg.norm(kernel_parameters - prev_parameters)
        if error < min_error:
            min_error = error
            min_kernel_parameters = prev_parameters
        iterations += 1
    return min_kernel_parameters, residual_errors

def identify(face_images, ms_eigen_pairs, kernel_parameters, num_nearest_neighbors, unidentified_image=None):
    
    if (unidentified_image == None):
        # Introduce new face and classify
        new_face_file = input("Enter the filename of an image to be classified: ")
        new_face = FaceImage(new_face_file, None)
    else:
        new_face = unidentified_image

    new_face_projection = projectToKernelSpace(new_face.image_vector, ms_eigen_pairs, face_images, kernel_parameters[0], kernel_parameters[1], kernel_parameters[2], kernel_parameters[3], kernel_parameters[4])

    corresponding_faces = KNearestNeighbors(face_images, new_face_projection, num_nearest_neighbors)
    print("Closest Faces:")
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
    # Optimal Eigenspace dimension
    OPTIMAL_DIMENSION = 7
    # Optimal nearest neighbors to consider.
    OPTIMAL_NEAREST_NEIGHBORS = 3
    # Import training data set.
    face_images = importDataSet()

    # KERNEL_PARAMETERS takes the form [alpha, kernel_dimension, kernel_offset, beta, sigma] 
    KERNEL_PARAMETERS, RESIDUAL_ERRORS = optimize_kernel(face_images, OPTIMAL_DIMENSION)
    # (Equivalent to Eigenfaces): KERNEL_PARAMETERS = [1, 1, 0, 1, 1]
    # (Graident Descent Optimized) - Residual Error of 1.9146217702328295e-08: KERNEL_PARAMETERS = [-2.48516865e+09, -2.03145406e+11, -5.29395592e-07, -2.48860032e+04, -2.69645074e+08]

    # Calculate K matrix
    K = kernelMatrix(face_images, KERNEL_PARAMETERS[0], KERNEL_PARAMETERS[1], KERNEL_PARAMETERS[2], KERNEL_PARAMETERS[3], KERNEL_PARAMETERS[4])

    # Normalize Kernel Matrix
    K = normalizeKernel(K)

    # Calculate eigen vectors and values from the Normalized Kernel Matrix.
    eigen_values, eigen_vectors = np.linalg.eig(K)

    # Pair the eigenvectors and eigenvalues then order pairs by decreasing eigenvalue magnitude.
    eigen_pairs = []
    for i in range(len(eigen_values)):
        eigen_pairs.append(EigenPair(eigen_values[i], eigen_vectors[i]))
    eigen_pairs.sort(key=lambda x: x.magnitude, reverse=True)

    # Choose a subset of eigenpairs corresponding to OPTIMAL_DIM largest eigenvalues.
    ms_eigen_pairs = []
    for k in range(OPTIMAL_DIMENSION):
        ms_eigen_pairs.append(eigen_pairs[k])

    # Classify the given training dataset based on the chosen subspace.
    for face in face_images:
        face.OMEGA_k = projectToKernelSpace(face.image_vector, ms_eigen_pairs, face_images, KERNEL_PARAMETERS[0], KERNEL_PARAMETERS[1], KERNEL_PARAMETERS[2], KERNEL_PARAMETERS[3], KERNEL_PARAMETERS[4])
        print(f"Face: {face.identity} has OMEGA_K: {face.OMEGA_k}")

    unidentified_images = importDataSet(os.getcwd() + "/Face_Images/unidentified", True)
    performance_vector = []
    for unidentified_image in unidentified_images:
        performance_vector.append(identify(face_images, ms_eigen_pairs, KERNEL_PARAMETERS, OPTIMAL_NEAREST_NEIGHBORS, unidentified_image))
    print(f"Using Kernel Parameters: {KERNEL_PARAMETERS}")
    print(f"The resulting algorithm achieves {(sum(performance_vector)/len(performance_vector))*100}% recognition accuracy with a residual error of {getError(face_images, KERNEL_PARAMETERS, OPTIMAL_DIMENSION)}.")

    plt.figure(1)
    plt.title("Residual Error vs. Iterations as Performing Gradient Descent")
    plt.plot(range(0, len(RESIDUAL_ERRORS)), RESIDUAL_ERRORS)

    plt.figure(2)
    plt.title("Residual Error vs. Iterations (3+) as Performing Gradient Descent")
    plt.plot(range(2, len(RESIDUAL_ERRORS)), RESIDUAL_ERRORS[2:])
    plt.show()

if __name__ == "__main__":
    main()