## MTHE 493 Thesis Project
### Face_Images
Dataset of face images from https://www.kaggle.com/gasgallo/faces-data-new/data.
#### faces_dataset
The default folder to be used as the dataset of faces for the algorithm to train on which contains all of the individuals that the algorithm may identify.
#### unitentified
The default folder to be used as the dataset of faces which are yet to be identified.
### Utilities.py
Methods that are common among all recognition algorithms.
#### Class FaceImage:
A class which describes a face image with a 150x150 pixel array representation, a 22500x1 vector representation, an integer valued identity, and an Omega vector which represents its "location" in the low dimensional image space.
#### Class EigenPair:
A class which describes a eigenvector/value pair for the purpose of sorting eigenvectors based on the eigenvalue magnitude.
#### def importDataSet(foldername = os.getcwd() + "/Face_Images/faces_dataset"):
A method to import the dataset of face images as defined in Face_Images/faces_dataset and create an array of FaceImage classes.
#### def euclideanDistance(vector_1, vector_2)
A method to calculate the euclidean distance between two given FaceImage's Omega vectors.
### Eigenfaces.py
Python implementation of "Eigenfaces for recognition" (1991) which uses PCA to analyze a predefined dataset of face images and then match a newly introduced face image to an individual in the data set.