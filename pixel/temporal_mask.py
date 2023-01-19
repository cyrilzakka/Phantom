# Copyright (c) Cyril Zakka.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import csv
import getpass
import pathlib
import argparse
import multiprocessing
from multiprocessing import Pool, set_start_method

import cv2
import numpy as np
import pydicom as dicom
from pydicom.pixel_data_handlers.util import convert_color_space

class TemporalMaskingPipeline:
    headers = ['dcm_name', 'patient_id', 'accession_number', 'num_frames', 'fps', 'avi_path']
    default_fps = 30.0
    default_len = 1
    """
    As the name suggests, the TemporalMasking pipeline leverages the temporal aspect of a dynamic
    modality (eg echocardiography) to mask any burnt-in (static) pixels.

    Args:
        source_folder: Source folder to find all the DICOMs
        destination_folder: Destination path for converted DICOMs.
        target_size: Target size to resize DICOM
        use_color: Specify whether output videos are in RGB or greyscale
    """
    def __init__(self, source_folder, destination_folder, target_size=224, use_color=True):
        self.destination_folder = pathlib.Path(destination_folder)
        self.destination_folder.mkdir(parents=True, exist_ok=True)

        self.source_folder = pathlib.Path(source_folder)
        # Check if source folder exists
        if not self.source_folder.exists():
            raise FileNotFoundError(f"Source folder {str(self.source_folder)} does not exist.")

        self.csv_ledger = (
            self.destination_folder / f'{self.source_folder.name}.csv'
        )
        print(f"Writing ledger to {str(self.csv_ledger)}")

        self.target_size = target_size
        self.use_color = use_color
        self.attrs = {}

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

    def _load_dcm(self, dicom_path: str):
        """
        General function to extract video and attributes of interest from DICOM
        Args:
            dicom_path: Path pointing to the DICOM.
        Returns:
            Returns numpy array (T,H,W,C)
        """
        ds = dicom.dcmread(dicom_path)
        dicom_meta = self._extract_meta(ds)

        try:
            media_array = convert_color_space(ds.pixel_array, ds.PhotometricInterpretation, 'RGB')
        except:
            try:
                media_array = ds.pixel_array
            except:
                # Invalid DICOM
                return None, None

        # Check whether DICOM is a static image
        if len(media_array.shape) < 4 or media_array.shape[0] < 10:
            print(f"NotImplementedError: Single DICOM images are not supported.")
            return None, None

        dicom_meta['dcm_name'] = dicom_path.name
        return media_array, dicom_meta  # (T, H, W, C)

    def _extract_meta(self, ds):
        """
        Helper function to extract DICOM attributes of interest
        Args:
            ds: DICOM object.

        Returns:
            Dictionary containing extracted information.
        """
        meta_dict = {
            "sop_uid": ds.get([0x0008, 0x0018]), # Uniquely identifies the referenced SOP Instance
            "accession_number": ds.get([0x0008, 0x0050]), # A RIS generated number that identifies the order for the Study.
            "recommended_fps": ds.get([0x0008, 0x2144]), # 30
            "fps": ds.get([0x0018, 0x0040]), # 30
            "num_frames": ds.get([0x0028, 0x0008]), # 91
            "height": ds.get([0x0028, 0x0010]),
            "width": ds.get([0x0028, 0x0011]),
            "series_uid": ds.get([0x0020, 0x000e]), # User or equipment generated Study identifier
            "patient_id": ds.get([0x0010, 0x0020]), # RF5251224
        }
        meta_dict['fps'] = (
            meta_dict['fps'].value if meta_dict['fps'] is not None else self.default_fps
        )
        return meta_dict

    def _generate_mask(self, array):
        """
        Generate mask for DICOM by obtaining the static pixels from the ultrasound.
        Args:
            array: DICOM pixel array of shape (T,H,W,C).
        Returns:
            DICOM mask (array).
        """
        f0 = cv2.cvtColor(array[0], cv2.COLOR_RGB2GRAY)
        f1 = cv2.cvtColor(array[1], cv2.COLOR_RGB2GRAY)
        mask_static = cv2.bitwise_and(f0, f1)
        for image in array[2:]:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask_static = cv2.bitwise_and(mask_static, image)
        mask_static[mask_static > 0] = 1

        mask = (np.max(array, axis = 0) - np.min(array, axis = 0)).max(axis = 2) == 0

        merged_mask = mask & mask_static
        return merged_mask

    def _crop_dcm(self, array, meta):
        """
        Crop and square DICOM array.
        Args:
            array: DICOM pixel array of shape (T,H,W,C).
        Returns:
            Cropped pixel array.
        """
        # Square crop
        diff = abs(array.shape[1] - array.shape[2]) // 2
        if diff > 0:
            if array.shape[1] > array.shape[2]:
                array = array[:, diff:-diff, :, :]
            else:
                array = array[:, :, diff:-diff, :]
        return array

    def _convert_dcm_to_avi(self, array, meta):
        """
        Convert DICOM to AVI.

        Args:
            array: DICOM pixel array of shape (T,H,W,C)  
        Retuns:
            None      
        """
        filename = meta['series_uid'].value + '_' + meta['sop_uid'].value
        video_filename = os.path.join(self.destination_folder, filename + '.avi')
        crop_size = (self.target_size, self.target_size)
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        fps = meta['fps']
        out = cv2.VideoWriter(video_filename, fourcc, fps, crop_size)
        array = self._crop_dcm(array, meta)
        mask = self._generate_mask(array)
        for i in range(meta['num_frames'].value):
            channel = array[i,:,:,0] if not self.use_color else array[i,:,:,:]
            channel[mask > 0] = 0
            output = cv2.resize(channel, crop_size, interpolation=cv2.INTER_NEAREST)

            if self.use_color:
                avi = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR
                out.write(avi)
            else:
                avi = cv2.merge([output ,output, output])
                out.write(avi)
        out.release()

        # Write video path to csv
        meta['avi_path'] = video_filename

    @staticmethod
    def _get_ledger_row(meta):
        return {
            'dcm_name': meta['dcm_name'],
            'patient_id': meta['patient_id'].value,
            'accession_number': meta['accession_number'].value,
            'num_frames': meta['num_frames'].value,
            'fps': meta['fps'],
            'avi_path': meta['avi_path'],
        }

    def _process_dcm(self, path):
        """
        Helper function to fully process a single DICOM
        Args:
            path: Path to DICOM
        """
        dcm, meta = self._load_dcm(path)
        if dcm is not None:
            self._convert_dcm_to_avi(dcm, meta)
            return meta  # return metaata dict so we can write in master process

        return None  # explicitly indicate we've failed

    def preprocess(self):
        """
        Main function for preprocessing. Supports multithreading.
        """
        print("Looking for all DICOMs...")
        dcms = sorted(self._find_dcms(self.source_folder))
        print(f"Found {len(dcms)} in specified path: {self.source_folder}")

        # Initialize ledger file if necessary
        ledger_exists = self.csv_ledger.is_file()
        ledger_kwargs = {'delimiter': ',', 'lineterminator': '\n', 'fieldnames': self.headers}
        if not ledger_exists:
            with open(self.csv_ledger, 'w') as f:
                writer = csv.DictWriter(f, **ledger_kwargs)
                writer.writeheader()

        with Pool(multiprocessing.cpu_count()-1) as pool, open(self.csv_ledger, 'a') as f:
            writer = csv.DictWriter(f, **ledger_kwargs)
            for meta in pool.imap_unordered(self._process_dcm, dcms):
                if meta is None:
                    continue
                writer.writerow(self._get_ledger_row(meta))
                print(f'Processed and documented entry {meta["dcm_name"]}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '-s', '--source', help='Source folder', type=pathlib.Path,
        default=pathlib.Path(f'/scratch/users/{getpass.getuser()}/dicom/'),
    )
    p.add_argument(
        '-d', '--destination', help='Destination folder', type=pathlib.Path,
        default=pathlib.Path(f'/scratch/users/{getpass.getuser()}/processed/'),
    )
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    set_start_method("spawn")
    tmppipeline = TemporalMaskingPipeline(ARGS.source, ARGS.destination)
    tmppipeline.preprocess()
    print("Done")