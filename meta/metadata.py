# Copyright (c) Cyril Zakka.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import pathlib
import argparse
import dateutil.parser
from uuid import uuid4
import multiprocessing
from multiprocessing import Pool
from datetime import datetime, timedelta

import pandas as pd
import pydicom as dicom
from omegaconf import DictConfig, OmegaConf

class MetadataPurgePipeline:

    image_data = ['PixelData', 'Rows', 'Columns', 'BitsStored', 'BitsAllocated', 'PixelRepresentation', 'SamplesPerPixel', 'PhotometricInterpretation']
    """
    The MetadataPurge pipeline removes all metadata from DICOMs."""
    def __init__(self, cfg: DictConfig, source_folder: str, destination_folder: str = None):

        self.conf = OmegaConf.load(cfg)
        self.has_base = self.conf.get('defaults', None)
        self.keep = OmegaConf.to_object(self.conf.keep)
        self.generate = OmegaConf.to_object(self.conf.generate)
        self.jitter = OmegaConf.to_object(self.conf.jitter)

        # Check if base config exists
        if self.has_base is not None:
            alt_path = pathlib.Path(cfg).parent / f'{self.conf.defaults}.yaml'
            self.alt_conf = OmegaConf.load(alt_path)
            self.keep += OmegaConf.to_object(self.alt_conf.keep)
            self.generate += OmegaConf.to_object(self.alt_conf.generate)
            self.jitter.update({k: v for k, v in OmegaConf.to_object(self.alt_conf.jitter).items() if not k in self.jitter})


        if destination_folder is not None:
            self.in_place = False
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

    def _clean_dcm(self, dcm_path):
        """
        Function to clean a single DICOM.

        Args:
            dcm_path: Path to DICOM to clean.
        """
        # Load DICOM
        dcm = dicom.dcmread(dcm_path)

        # Transfer tags
        all_keep = self.keep + self.generate + list(self.jitter.keys())
        data_elem = []
        for tag in set(all_keep):
            de = dcm.data_element(tag)
            data_elem.append(de)

        for image_tag in self.image_data:
            data_elem.append(dcm.data_element(image_tag))

        # Clear dcm
        dcm.clear()

        # Add elements
        for elem in data_elem:
            if elem is not None:
                dcm.add(elem)

        # Jitter tags
        for tag, value in self.jitter.items():
            if tag in dcm:
                frmtstr = '%Y%m%d' if dcm[tag].VR == 'DA' else '%Y%m%d%H%M%S'
                dcm[tag].value = (pd.to_datetime(dcm[tag].value, infer_datetime_format=True) + timedelta(days=value)).strftime(frmtstr)
        
        # Anonymize tags
        for tag in self.generate:
            if tag in dcm:
                dcm[tag].value = str(uuid4())

        # Write to destination
        if self.in_place:
            dcm.save_as(dcm_path)
        else:
            dcm.save_as(self.destination_folder / dcm_path.name)


    def clean(self):
        """
        Function to clean all DICOMs in source folder.
        """
        # Find all DICOMs
        dcms = self._find_dcms(self.source_folder)
        print(f"Found {len(dcms)} DICOMs.")

        # Create a pool of workers
        pool = Pool(processes=multiprocessing.cpu_count())
        pool.map(self._clean_dcm, dcms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    parser.add_argument('--source', type=str, default=pathlib.Path(f'/Users/cyril/Desktop/dcm'))
    parser.add_argument('--destination', type=str, default=pathlib.Path(f'/Users/cyril/Desktop/cleaned'))
    args = parser.parse_args()

    pipeline = MetadataPurgePipeline(args.config, args.source, args.destination)
    pipeline.clean()