from deforestation_detection import train_and_evaluate
from itertools import product
from multiprocessing import Pool, cpu_count
import argparse


def run_task(params):
    method = params["method"]
    n_seg = params["n_seg"]
    result_path = params["result_path"]
    superpixels_root = params["superpixels_root"]
    
    train_and_evaluate(method, n_seg, [
                       3, 2, 1], 42, 'scenes_allbands_ndvi', 'truth_masks', result_path, superpixels_root)
    return f"Finished {method} with {n_seg} segments!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run deforestation detection with different superpixel methods.')
    parser.add_argument('--result-path', '-r', type=str,
                        default='results', help='Path to save the results')
    parser.add_argument("--superpixels-root", type=str)
    parser.add_argument('--num-processes', '-n', type=int)
    args = parser.parse_args()

    # Define parameters
    methods = [
        "AINET"
        "CRS",
        "DISF",
        "DRW",
        "ERGC",
        "ERS",
        "ETPS",
        "GMMSP",
        "GRID",
        "IBIS",
        "ISF",
        "LSC",
        "ODISF",
        "RSS",
        "SCALP",
        "SEEDS",
        "SH",
        "SICLE",
        "SIN",
        "SLIC",
        "SNIC",
        "SSFCN"
    ]

    n_seg_parameters = [1000, 1500, 2000, 2500, 3000,
                        4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # Choose the number of processes to run simultaneously
    num_processes = args.num_processes or 4

    task_parameters = [
        {
            "method": method,
            "n_seg": n_seg,
            "result_path": args.result_path,
            "superpixels_root": args.superpixels_root
        }
        for method, n_seg in product(methods, n_seg_parameters)
    ]

    # Create a pool of workers with specified number of processes
    with Pool(processes=num_processes) as pool:
        # Map the function to the parameters and execute in parallel
        results = pool.map(run_task, task_parameters)

    for result in results:
        print(result)

    print("Finished all jobs!")
