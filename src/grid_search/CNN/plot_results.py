import sys

import yaml
import git
import pandas as pd
PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)

from src.utils import utils

if __name__== "__main__":
    initial_config_path = PATH_TO_ROOT + "/src/grid_search/CNN/grid_search.yaml"
    grid_search_initials = utils.get_config(initial_config_path)

    # For first grid search
    grid_search_results = utils.get_config(PATH_TO_ROOT+'/results/cnn_grid_search/results_kernel+filter.yaml')
    df_results = pd.DataFrame(grid_search_results)
    
    # Convert list values in the "Filter Size" column to tuples
    df_results["Filter Size"] = df_results["Filter Size"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    
    # Pivot the DataFrame
    heatmap_data = df_results.pivot(index='Kernel Size', columns='Filter Size', values='CV Accuracy')
    
    # Plot the heatmap
    filename="heatmap_grid_search_kf.png"
    utils.plot_grid_heatmap(heatmap_data, config_path=initial_config_path, filename=None)

    # For second grid search
    grid_search_results = utils.get_config(PATH_TO_ROOT+'/results/cnn_grid_search/results_epochs+lr.yaml')
    df_results = pd.DataFrame(grid_search_results)
    
    # Pivot the DataFrame
    heatmap_data = df_results.pivot(index='Learning Rate', columns='Epochs', values='CV Accuracy')
    
    # Plot the heatmap
    utils.plot_grid_heatmap(heatmap_data, config_path=initial_config_path, filename="heatmap_grid_search_elr.png")


    
