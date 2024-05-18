from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from classification_dataset import ClassificationDataset
import os


def train_and_evaluate(method, n_segs_param, band_combination, seed, images_path, truth_path, result_path, superpixels_root):
    clf = SVC(C=100,
              gamma='scale',
              kernel='rbf',
              class_weight='balanced',
              random_state=seed)

    if isinstance(n_segs_param, int):
        n_segs_param = str(n_segs_param)

    superpixels_path = os.path.join(
        superpixels_root, method, 'scenes_pca', n_segs_param)

    train_regions = ['x01', 'x02', 'x06', 'x07', 'x08', 'x09', 'x10']
    test_regions = ['x03', 'x04']

    train_dataset = ClassificationDataset(
        images_path, truth_path, superpixels_path, train_regions, band_combination)
    X_train, y_train = train_dataset.get_set()

    test_dataset = ClassificationDataset(
        images_path, truth_path, superpixels_path, test_regions, band_combination)
    X_test, y_test = test_dataset.get_set()

    band_combination_name = 'B' + \
        'B'.join([str(band) for band in band_combination])
    if band_combination == [3, 2, 1]:
        band_combination_name = 'rgb'

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    tp = sum([1 for y, y_pred in zip(y_test, y_pred) if y == 0 and y_pred == 0])
    tn = sum([1 for y, y_pred in zip(y_test, y_pred) if y == 1 and y_pred == 1])
    fp = sum([1 for y, y_pred in zip(y_test, y_pred) if y == 1 and y_pred == 0])
    fn = sum([1 for y, y_pred in zip(y_test, y_pred) if y == 0 and y_pred == 1])

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Save results to a txt file
    os.makedirs(f'{result_path}', exist_ok=True)
    with open(f"{result_path}/{method}_{n_segs_param}_{band_combination_name}.txt", "w") as f:
        f.write(f"bal_acc: {bal_acc}\n")
        f.write(f"specificity: {specificity}\n")
        f.write(f"sensitivity: {sensitivity}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"method: {method}\n")
        f.write(f"n_segs_param: {n_segs_param}\n")
        f.write(f"band_combination: {band_combination_name}\n")
        f.write(f"train_size: {len(X_train)}\n")
        f.write(f"train_forest_count: {len(y_train) - sum(y_train)}\n")
        f.write(f"train_non_forest_count: {sum(y_train)}\n")
        f.write(f"test_size: {len(X_test)}\n")
        f.write(f"test_forest_count: {len(y_test) - sum(y_test)}\n")
        f.write(f"test_non_forest_count: {sum(y_test)}\n")

    print('Done!')
