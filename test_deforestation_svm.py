from svm import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate('CRS', '300', [3, 2, 1], 42, 'scenes_allbands_ndvi', 'truth_masks')