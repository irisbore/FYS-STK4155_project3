import os
import sys
import yaml
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import git
PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)


"""
This script contains functions for retrieving config, as well as various plotting functions
"""

def get_config(path) -> Dict:
    """
    Reads yaml config at path and returns it as python dictionary
    param str path: path to config file
    return: config
    rtype: dict()
    """

    config = None
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Creates a parser to read possible path to config from command line
    return: parser
    rtype: argparse.ArgumentParser()
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        help='Path to the config file')
    return parser

def get_config_path(default_path=None) -> str:
    """
    Fetches config path from either command line or default path.
    return: path to config
    rtype: str
    """

    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if parsed_args.config_path:
        config_path = parsed_args.config_path
    else:
        config_path = default_path
        print('Using default config')
    return config_path

def plot_grid_heatmap(df: pd.DataFrame, config_path:str, filename=None, title: str=None, annot = True):
    config = get_config(config_path)
    if config['grid_search'] == 'number_of_conv_layers':
        title = rf"CV Accuracy for Different Numbers of Layers"

    if config['grid_search'] == 'kernel+filter':
        lr = config["learning_rate"]
        title = rf"CV Accuracy for Number of Kernels and Filters, with $\eta = {lr} $"

    if config['grid_search'] == 'padding_vs_pooling':
        title = rf"CV Accuracy with and without Padding and Pooling"

    if config['grid_search'] == 'dropout_vs_activations':
        title = rf"CV Accuracy for Dropout Rates and Activation Functions (20 Epochs)"
    plt.clf()
    sns.heatmap(df, annot=annot, fmt='.3g')
    plt.title(title)
    plt.tight_layout()
    if config["save_plot"]:
        plt.savefig(PATH_TO_ROOT+config["save_path"]+'/'+filename)
    else:
        plt.show()

def plot_classwise(score_dict: dict, model: str, save_plot:bool):
    labels = list(score_dict.keys())
    values = list(score_dict.values())
    plt.figure(figsize=(10, 6))
    plt.plot(labels, values, marker='o', linestyle='-', color='b', markersize=8)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Classwise Accuracy on Testset for {model}', fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    if save_plot: 
        plt.savefig(f'{PATH_TO_ROOT}/results/evaluation/{model}_classwise_acc.png')
    else:
        plt.show()

def plot_parameter_study(param_values, accuracies, param_name, xticks=False):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, accuracies, 'bo-')
    plt.xlabel(param_name)
    if xticks: 
        plt.xticks(param_values)
    plt.ylabel('CV Accuracy (%)')
    plt.title(f'Impact of {param_name} on Model Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{PATH_TO_ROOT}/results/logreg/{param_name.lower().replace(" ", "_")}_study.png')
    plt.close()

