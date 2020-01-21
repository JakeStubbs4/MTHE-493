class EigenPair:
    eigen_value = None
    magnitude = None
    eigen_vector = None

    def __init__(self, eigen_value, eigen_vector):
        self.eigen_value = eigen_value
        self.magnitude = abs(eigen_value)
        self.eigen_vector = eigen_vector
