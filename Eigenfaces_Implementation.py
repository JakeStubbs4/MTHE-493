# MTHE-493 Facial Recognition Project
# EigenFaces Implementation
# Jake Stubbs

from matplotlib import pyplot as plt
import numpy as np
from Utilities import euclideanDistance, importDataSet, FaceImage, EigenPair

# Computes the vector representation of the average face of all of the faces in the provided dataset.
def averageVector(face_images):
    face_images_arrays = []
    for image in face_images:
        face_images_arrays.append(image.image_array)
    return np.mean(face_images_arrays, axis=0).reshape(-1,1)

# Computes the standard deviation of each face image and returns an array of deviation vectors.
def standardDeviation(face_images, average_face):
    face_deviations = []
    for face in face_images:
        face_deviations.append(face.image_vector - average_face)
    return face_deviations

# Computes the eigenvectors of the provided empirical covariance matrix A.
def covarianceEigenvectors(face_deviations, A):
    L = np.dot(np.transpose(A),A)
    return np.linalg.eig(L)

# Converts eigen vector to face images to be displayed.
def getEigenFace(eigen_vector, A):
    eigen_face = np.dot(A, eigen_vector).reshape(150,150)
    return eigen_face

# Projects newly introduced face image onto predetermined low dimensional image space.
def projectImage(face_image, eigen_pairs, average_face, A):
    projection = []
    for pair in eigen_pairs:
        omega_k = float(np.dot(np.dot(A, pair.eigen_vector), face_image - average_face))
        projection.append(omega_k)
    return projection

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

# To optimize Dimension of extracted linear subspace as well as K in KNN computation.
'''def optimizeDimension():
    unidentified_faces = importDataSet(os.getcwd() + "/unidentified")

    correct_identifications = dict()
    for i in range(1,8):
        correct_identifications[i] = 0
    print(correct_identifications)
    for i in range(1,8):
        # Number of neighbors to consider in KNN classification.
        K = i
        # Number of dimensions to use to construct subspace.
        DIM = 7

        # Choose a subset of eigenpairs corresponding to DIM largest eigenvalues. 
        ms_eigen_pairs = []
        for k in range(DIM):
            ms_eigen_pairs.append(eigen_pairs[k])

        # Classify the given training dataset based on the chosen subspace.
        for face in face_images:
            face.OMEGA_k = projectImage(face.image_vector, ms_eigen_pairs, average_face, A)
            print(face.OMEGA_k)

        for new_face in unidentified_faces:
            # Introduce new face and classify
            new_face_projection = projectImage(new_faced.image_vector, ms_eigen_pairs, average_face, A)

            corresponding_faces = KNearestNeighbors(face_images, new_face_projection, K)
            for face in corresponding_faces:
                print(face.identity)

            corresponding_face = classifyImage(corresponding_faces, new_face_projection)

            plt.figure(1)
            io.imshow(new_face.image_array)

            plt.figure(2)
            io.imshow(corresponding_face.image_array)

            #plt.show()
            plt.draw()
            plt.pause(1) # <-------
            plt.close()

            while True:
                try:
                    n = input("Enter a 1 if the previous identification was correct or a 0 if it was not:")
                    n = int(n)
                    break
                except ValueError:
                    print("Try again")
            correct_identifications[i] += n
            print(correct_identifications)

    for key in correct_identifications:
        correct_identifications[key] = correct_identifications[key]/len(unidentified_faces)
    lists = sorted(correct_identifications.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.xlabel("Dimension of Subspace")
    plt.ylabel("Identification accuracy")
    plt.show()'''

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
    
    '''TO INTRODUCE A SINGLE FACE AT A TIME:'''
    # Optimal dimension for accuracy of recognition.
    OPTIMAL_DIM = 7
    # Optimal nearest neighbors to consider for accuracy of recognition.
    OPTIMAL_K = 3
    # Choose a subset of eigenpairs corresponding to DIM largest eigenvalues. 
    
    ms_eigen_pairs = []
    for k in range(OPTIMAL_DIM):
        ms_eigen_pairs.append(eigen_pairs[k])

    # Classify the given training dataset based on the chosen subspace.
    for face in face_images:
        face.OMEGA_k = projectImage(face.image_vector, ms_eigen_pairs, average_face, A)
        print(face.OMEGA_k)

    # Introduce new face and classify
    new_face_file = input("Enter the filename of an image to be classified: ")
    new_face = FaceImage(new_face_file, None)

    new_face_projection = projectImage(new_face.image_vector, ms_eigen_pairs, average_face, A)

    corresponding_faces = KNearestNeighbors(face_images, new_face_projection, OPTIMAL_K)
    for face in corresponding_faces:
        print(face.identity)

    corresponding_face = classifyImage(corresponding_faces, new_face_projection)

    # TODO: Add some check which will determine if the match is close enough.
    #new_face.identity = corresponding_face[0].identity

    plt.figure(1)
    new_face.displayImage()

    plt.figure(2)
    corresponding_face.displayImage()

    plt.show()
    
main()