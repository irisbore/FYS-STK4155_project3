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
    if config['grid_search'] == 'kernel+filter':
        lr = config["learning_rate"]
        title = rf"CV Accuracy for Number of Kernels and Filters, with $\eta = {lr} $"

    if config['grid_search'] == 'epochs+lr':
        lr = config["learning_rate"]
        title = rf"CV Accuracy for Different Values of Epochs and Learning Rates"
    sns.heatmap(df, annot=annot)
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(config["save_path"]+'/'+filename)
    else:
        plt.show()


