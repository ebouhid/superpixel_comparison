import mahotas
import numpy as np

def get_haralick_features(image):
    features = []
    for channel in range(image.shape[2]):
        features.append(mahotas.features.haralick(image[:, :, channel]))
    return np.array(features)

