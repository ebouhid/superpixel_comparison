import mahotas
import numpy as np

def get_haralick_features(image):
    features = []
    for channel in range(image.shape[2]):
        try:
            features.append(mahotas.features.haralick(image[:, :, channel]))
        except ValueError as e:
            print(f"Error in channel {channel}")
            print(f"Image shape: {image.shape}")
            print(f"Unique values: {np.unique(image[:, :, channel])}")
            features.append(np.zeros((4, 13)))
    return np.array(features)

