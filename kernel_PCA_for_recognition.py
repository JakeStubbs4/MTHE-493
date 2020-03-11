# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from matplotlib import pyplot as plt
from random import randrange
import numpy as np
import math
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

# Apply a linear combination of the Polynomial kernel with parameters d=dimension and c=offset and the gaussian kernel with parameter Sigma to data points Xi and Xj.
def applyKernel(Xi, Xj, alpha, d, c, beta, Sigma):
    result = alpha*float((np.dot(np.transpose(Xi), Xj) + c)**d) + beta*float(math.exp(-1*((np.linalg.norm(Xi - Xj)**2)/(2*(Sigma**2)))))
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
def projectToKernelSpace(current_face_image, eigen_vectors, face_images, alpha, d, c, beta, Sigma):
    projection = eigen_vectors[0].eigen_vector*applyKernel(current_face_image, face_images[0].image_vector, alpha, d, c, beta, Sigma)
    for i in range(1, len(eigen_vectors)):
        projection += eigen_vectors[i].eigen_vector*applyKernel(current_face_image, face_images[i].image_vector, alpha, d, c, beta, Sigma)
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
    kernel_Sigma = 0.000000000000001
    kernel_parameters = [kernel_alpha, kernel_dimension, kernel_offset, kernel_beta, kernel_Sigma]
    print(f"INITIAL KERNEL PARAMETERS: {kernel_parameters}")
    delta = 0.00000001
    precision = 0.001
    accuracy = precision + 1
    max_iterations = 1000
    iterations = 1
    da = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [delta, 0, 0, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    dd = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, delta, 0, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    dc = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, delta, 0, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    db = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, 0, delta, 0])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    ds = lambda x : (getError(face_images, [sum(x) for x in zip(kernel_parameters, [0, 0, 0, 0, delta])], eigenspace_dimension) - getError(face_images, kernel_parameters, eigenspace_dimension))/delta
    while accuracy > precision and iterations < max_iterations:
        prev_parameters = kernel_parameters
        cost_vector = [da(prev_parameters), dd(prev_parameters), dc(prev_parameters), db(prev_parameters), ds(prev_parameters)]
        error = getError(face_images, prev_parameters, eigenspace_dimension)
        print(f"CURRENT RESIDUAL ERROR IS: {np.format_float_positional(error)}")
        learning_rate_vector = np.multiply(1/iterations, [1/(abs(cost_vector[0]) + delta), 1/(abs(cost_vector[1]) + delta), 1/(abs(cost_vector[2]) + delta), 1/(abs(cost_vector[3]) + delta), 1/(abs(cost_vector[4]) + delta)])
        kernel_parameters = prev_parameters - np.multiply(learning_rate_vector, cost_vector)
        accuracy = np.linalg.norm(kernel_parameters - prev_parameters)
        iterations += 1
        if iterations % 50 == 0 or iterations < 25:
            print(f"At iteration {iterations} the kernel parameters are: {kernel_parameters}, accuracy is {accuracy}")
    return kernel_parameters

def identify(face_images, kernel_parameters, eigenspace_dimension, num_nearest_neighbors):
    # Calculate K matrix
    K = kernelMatrix(face_images, kernel_parameters[0], kernel_parameters[1], kernel_parameters[2], kernel_parameters[3], kernel_parameters[4])

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
        face.OMEGA_k = projectToKernelSpace(face.image_vector, ms_eigen_pairs, face_images, kernel_parameters[0], kernel_parameters[1], kernel_parameters[2], kernel_parameters[3], kernel_parameters[4])

    # Introduce new face and classify
    new_face_file = input("Enter the filename of an image to be classified: ")
    new_face = FaceImage(new_face_file, None)

    new_face_projection = projectToKernelSpace(new_face.image_vector, ms_eigen_pairs, face_images, kernel_parameters[0], kernel_parameters[1], kernel_parameters[2], kernel_parameters[3], kernel_parameters[4])

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
    OPTIMAL_NEAREST_NEIGHBORS = 2
    # Import training data set.
    face_images = importDataSet()

    KERNEL_PARAMETERS = optimize_kernel(face_images, OPTIMAL_DIMENSION)
    # DECENT RESULT 1: KERNEL_PARAMETERS = [8.74277761e-01, -4.20504505e-04,  9.56852310e-06,  1.17068664e-04, 1.00000000e-28]
    identify(face_images, KERNEL_PARAMETERS, OPTIMAL_DIMENSION, OPTIMAL_NEAREST_NEIGHBORS)


if __name__ == "__main__":
    main()