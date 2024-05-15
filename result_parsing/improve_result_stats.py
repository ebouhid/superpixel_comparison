import pandas as pd
import skimage.io as io
from PIL import Image
import numpy as np


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('results.csv')
    df = df[df["n_segs_param"] == 1000]
    
    for index, row in df.iterrows():
        method = row['method']
        n_segs_param = row['n_segs_param']

        # Load segmentation
        for region in ["x01", "x02", "x03", "x04", "x06", "x07", "x08", "x09", "x10"]:
            try:
                seg_path = f"SegmentationResults/{method}/scenes_rgb/{n_segs_param}/{region}.pgm"
                seg = io.imread(seg_path)
            except FileNotFoundError:
                seg_path = f"SegmentationResults/{method}/scenes_rgb/{n_segs_param}/{region}.png"
                seg = Image.open(seg_path)
                seg = np.array(seg)

            # Load truth
            truth_path = f"truth_masks/truth_{region}.npy"
            truth = np.load(truth_path)

            # Check if the shapes are the same
            assert seg.shape[:2] == truth.shape[:2]
            

