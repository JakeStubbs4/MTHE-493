## MTHE 493 Thesis Project
### Utilities.py
Methods that are common among all recognition algorithms.
#### FaceImage:
A class which describes a face image with a 150x150 array representation, a 22500x1 vector representation, an integer valued identity, and an Omega vector which represents its "location" in the low dimensional image space.
#### EigenPair:
A class which describes a eigenvector/value pair for the purpose of sorting eigenvectors based on the eigenvalue magnitude.
#### importDataSet:
A method to import the dataset of face images as defined in Face_Images/faces_dataset and create an array of FaceImage classes.
#### euclideanDistance:
A method to calculate the euclideanDistance between two given FaceImage's Omega vectors.
### Eigenfaces.py
Python implementation of "Eigenfaces for recognition" (1991) which uses PCA to analyze a predefined dataset of face images and then match a newly introduced face image to an individual in the data set.