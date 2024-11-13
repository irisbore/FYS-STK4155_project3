import os
import sys
import yaml
import argparse
import numpy as np
from typing import List, Tuple, Dict


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

