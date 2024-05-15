import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import skimage.io as io

def count_total_segs(method, n_segs_param):
    train_candidates = 0
    test_candidates = 0
    for region in ["x01", "x02", "x03", "x04", "x06", "x07", "x08", "x09", "x10"]:
        print(f"Counting superpixels for {method} with {n_segs_param} segments")
        try:
            seg_path = f"SegmentationResults/{method}/scenes_rgb/{n_segs_param}/rgb_{region}.pgm"
            seg = io.imread(seg_path)
        except FileNotFoundError:
            seg_path = f"SegmentationResults/{method}/scenes_rgb/{n_segs_param}/rgb_{region}.png"
            seg = Image.open(seg_path)
            seg = np.array(seg)
        if region in ["x03", "x04"]:
            test_candidates += len(np.unique(seg))
        else:
            train_candidates += len(np.unique(seg))
    return {"train_candidates": train_candidates, "test_candidates": test_candidates}

def export_latex_format(df, directory):
    # Group data by method
    grouped = df.groupby('method')
    
    # Iterate over each method
    for method, method_df in grouped:
        # Create a file for each method
        file_path = os.path.join(directory, f'{method}-scenes_rgb-BA.dat')
        with open(file_path, 'w') as file:
            file.write("Superpixels Score\n")
            for index, row in method_df.iterrows():
                file.write(f"{row['n_segs_param']} {row['bal_acc']}\n")
                
    print("Data exported in LaTeX format.")


def parse_txt_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            data[key.strip()] = value.strip()
    return data

def main(directory):
    # Get list of txt files in the directory
    txt_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    
    # Parse each file and store the data
    parsed_data = []
    for file_name in txt_files:
        file_path = os.path.join(directory, file_name)
        parsed_data.append(parse_txt_file(file_path))
    
    # Create DataFrame
    df = pd.DataFrame(parsed_data)
    
    # Reorder columns
    columns_order = ['method', 'n_segs_param'] + [col for col in df.columns if col not in ['method', 'n_segs_param']]
    df = df[columns_order]
    
    # Convert columns to appropriate types
    df['bal_acc'] = df['bal_acc'].astype(float)
    df['n_segs_param'] = df['n_segs_param'].astype(int)
    
    df = df.sort_values(by=['method', 'n_segs_param'], ascending=True)

    for index, row in df.iterrows():
        method = row['method']
        n_segs_param = row['n_segs_param']
        train_candidates, test_candidates = count_total_segs(method, n_segs_param).values()
        df.loc[index, 'train_candidates'] = train_candidates
        df.loc[index, 'test_candidates'] = test_candidates

    # df = df.rename(columns={
    #     "forest_acc": "specificity",
    #     "non_forest_acc": "sensitivity",
    # })
    
    # Save DataFrame to a CSV file (optional)
    df.to_csv('results.csv', index=False)
    
    # Generate line plot using Matplotlib
    plt.figure(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':']
    for i, (method, method_df) in enumerate(df.groupby('method')):
        line_style = line_styles[i % len(line_styles)]
        plt.plot(method_df['n_segs_param'], method_df['bal_acc'], label=method, linestyle=line_style)

    plt.title('Balanced Accuracy vs n_segs_param')
    plt.xlabel('n_segs_param')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    
    # Save plot as image
    plot_output_path = 'results.png'
    plt.savefig(plot_output_path)

    # Clear plot
    plt.clf()

    plt.figure(figsize=(10, 6))
    
    # Get unique methods
    unique_methods = df['method'].unique()
    
    # Assign random colors and markers to each method
    # Set a seed for reproducibility
    random.seed(43)
    random.shuffle(unique_methods)
    method_colors = {method: plt.cm.jet(i/len(unique_methods)) for i, method in enumerate(unique_methods)}
    method_markers = {method: random.choice(['o', 's', '^', 'x', 'D', 'P', '*']) for method in unique_methods}
    
    # Plot each method separately
    for method, method_df in df.groupby('method'):
        plt.scatter(method_df['n_segs_param'], method_df['bal_acc'], label=method, color=method_colors[method], marker=method_markers[method])

    plt.title('Balanced Accuracy vs n_segs_param')
    plt.xlabel('n_segs_param')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    
    # Save plot as image
    plot_output_path = 'scatter_results.png'
    plt.savefig(plot_output_path)
    
    return df, plot_output_path

if __name__ == "__main__":
    directory = "./results/"
    latex_dir = "./latex/"
    df, plot_output_path = main(directory)
    print("DataFrame:")
    print(df)
    
    # Export data in LaTeX format
    os.makedirs(latex_dir, exist_ok=True)
    export_latex_format(df, latex_dir)
