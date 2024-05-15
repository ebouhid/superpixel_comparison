import numpy as np
import skimage
from tqdm import tqdm
from haralick_features import get_haralick_features
from glob import glob
from PIL import Image

def get_hor(segment):
    # flattening segment
    segment = segment.flatten()

    NFP = np.where(segment == 2, 1, 0).sum()
    NP = segment.size
    NNP = NP - NFP

    HoR = max([NFP, NNP]) / NP

    return HoR

def get_major_class(mask):
    if np.argmax(np.bincount(mask.flatten().astype(np.uint8))) == 2:
        return "forest"
    elif np.argmax(np.bincount(mask.flatten().astype(np.uint8))) == 3:
        return "non_forest"
    else:
        return "non_forest"

def get_region(path):
    print(path)
    return f"{path.split('/')[-1].split('.')[0]}"

def find_superpixel_bounding_boxes(superpixel_matrix):
    """
    Finds bounding boxes for each superpixel segment in the given superpixel matrix.

    Args:
    - superpixel_matrix: A 2D matrix representing superpixels where each superpixel is marked with a unique label.

    Returns:
    - A list of tuples, each tuple representing the bounding box coordinates (x_min, y_min, x_max, y_max) for each segment.
    """
    bounding_boxes = []
    for label in np.unique(superpixel_matrix):
        if label == 0:  # Skip background label
            continue
        # Find all pixels with the current label
        indices = np.where(superpixel_matrix == label)
        x_min, y_min = np.min(indices, axis=1)
        x_max, y_max = np.max(indices, axis=1)
        bounding_boxes.append((x_min, y_min, x_max, y_max))
    return bounding_boxes

def evaluate_segment(segment):
    if segment.shape[0] * segment.shape[1] < 70:
        return False
    
    classification = get_major_class(segment)

    if  (get_hor(segment > 0.7)) and (classification in ["forest", "non_forest"]):
        return True

    return False


class ClassificationDataset:
    def __init__(self, images_path, truth_path, superpixels_path, regions, combination):
        self.X = []
        self.y = []
        self.combination = combination
        images = glob(f"{images_path}/*.npy")
        for path in images:
            region = get_region(path)
            if region not in regions:
                print(f'skipping {region}...')
                continue
            image = np.load(path).astype(np.uint8).transpose(1, 2, 0) # transpose do H x W x C
            image = image[:, :, combination]
            truth = np.load(f"{truth_path}/truth_{region}.npy")
            try:
                superpixels = skimage.io.imread(f"{superpixels_path}/pca_{region}.pgm")
            except FileNotFoundError:
                superpixels = Image.open(f"{superpixels_path}/pca_{region}.png")
                superpixels = np.array(superpixels)
            
            assert truth.shape[:2] == superpixels.shape[:2]
            assert truth.shape[:2] == image.shape[:2]

            for segment_bbox in tqdm(find_superpixel_bounding_boxes(superpixels), desc=f"Getting features for {region}"):
                x_min, y_min, x_max, y_max = segment_bbox
                img_segment = image[y_min:y_max, x_min:x_max]
                msk_segment = truth[y_min:y_max, x_min:x_max]
                if evaluate_segment(msk_segment):
                    segment_har = get_haralick_features(img_segment)
                    segment_har = segment_har.flatten()
                    if get_major_class(msk_segment) == "forest":
                        cls_value = 0
                    elif get_major_class(msk_segment) == "non_forest":
                        cls_value = 1
                    else:
                        continue
                    self.X.append(segment_har)
                    self.y.append(cls_value)      

    def get_set(self):
        return self.X, self.y

   