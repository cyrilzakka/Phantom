# Copyright (c) Cyril Zakka.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import csv
import uuid
import getpass
import pathlib
import argparse
import multiprocessing
from multiprocessing import Pool, set_start_method

import pydicom as dicom
from omegaconf import DictConfig, OmegaConf


class MetadataPurgePipeline:
    """
    The MetadataPurge pipeline removes all metadata from DICOMs."""
    def __init__(self, cfg: DictConfig, source_folder: str, destination_folder: str = None):

        self.conf = cfg

        if destination_folder:
            self.destination_folder = pathlib.Path(destination_folder)
            self.destination_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.in_place = True

        self.source_folder = pathlib.Path(source_folder)
        if not self.source_folder.exists():
            raise FileNotFoundError(f"Source folder {str(self.source_folder)} does not exist.")

        self.csv_ledger = (
            self.destination_folder / f'{self.source_folder.name}.csv'
        )
        print(f"Writing ledger to {str(self.csv_ledger)}")

    def _find_dcms(self, path):
        """
        Function to traverse folder recursively and retrieve all DICOMs

        Args:
            path: Path to parent folder containing all DICOMs.

        Returns:
            List containing all DICOMs paths.
        """
        files = []
        extensions = ('.dcm')
        for root, _, f_names in os.walk(path):
            for f in f_names:
                if f.endswith(extensions):
                    files.append(pathlib.Path(os.path.join(root, f)))
        return files

    def _parse_config(self, config):
        """
        Function to parse config file.

        Args:
            config: Path to config file.

        Returns:
            Dictionary containing all config parameters.
        """
        with open(config, 'r') as f:
            config = json.load(f)
        return config