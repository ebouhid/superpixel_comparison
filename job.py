from deforestation_detection import train_and_evaluate
from itertools import product

if __name__ == '__main__':
    # Define parameters
    methods = [
        "CRS",
        "DISF",
        "ERGC",
        "ERS",
        "ETPS",
        "GRID",
        "IBIS",
        "ISF",
        # "LNSNet",
        "LSC",
        "ODISF",
        "RSS",
        "SEEDS",
        "SH",
        "SICLE",
        "SIN",
        "SLIC",
        "SNIC",
        "SSFCN"
    ]

    n_seg_parameters = [25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for method, n_seg in product(methods, n_seg_parameters):
        train_and_evaluate(method, n_seg, [3, 2, 1], 42, 'scenes_allbands_ndvi', 'truth_masks')
        print(f"Finished {method} with {n_seg} segments!")
    
    print("Finished all jobs!")
