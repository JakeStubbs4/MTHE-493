import numpy as np
from skimage import io
from skimage import transform

class FaceImage:
    image_array = None
    image_vector = None
    identity = None
    OMEGA_k = None

    def __init__(self, image, identity):
        if isinstance(image, str):
            self.image_array = np.array(transform.resize(
                io.imread(image, as_gray=True), (150, 150)))
        else:
            self.image_array = image
        if isinstance(identity, str):
            identity = identity.split('_')[0]
            if identity == "unidentified":
                self.identity = None
            else:
                self.identity = int(identity)
        else:
            self.identity = identity
        self.image_vector = self.image_array.flatten().reshape(-1, 1)
        self.OMEGA_k = None

    def displayImage(self):
        io.imshow(self.image_array)
