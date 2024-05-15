from deforestation_detection import train_and_evaluate
from itertools import product
from multiprocessing import Pool, cpu_count

def run_task(params):
    method, n_seg = params
    train_and_evaluate(method, n_seg, [3, 2, 1], 42, 'scenes_allbands_ndvi', 'truth_masks')
    return f"Finished {method} with {n_seg} segments!"

if __name__ == '__main__':
    # Define parameters
    methods = [
        # "AINET"
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

    n_seg_parameters = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # Choose the number of processes to run simultaneously
    num_processes = 8

    # Create a pool of workers with specified number of processes
    with Pool(processes=num_processes) as pool:
        # Map the function to the parameters and execute in parallel
        results = pool.map(run_task, product(methods, n_seg_parameters))

    for result in results:
        print(result)

    print("Finished all jobs!")
