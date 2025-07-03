#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import h5py

REBUILD_DATA = True

class Denoising:
    def __init__(self, label_dir='SIDD_Medium_Raw/Data', img_size=256):
        self.label_dir = label_dir
        self.img_size = img_size
        self.training_data = []

    def _load_mat(self, path):
        """Try scipy for legacy .mat, fall back to h5py for v7.3."""
        try:
            mat = loadmat(path)
            # pick first real key (skip __header__, etc.)
            key = next(k for k in mat if not k.startswith('__'))
            return mat[key]
        except NotImplementedError:
            with h5py.File(path, 'r') as f:
                # pick first dataset that isn’t a group of metadata
                key = next(k for k in f.keys() if isinstance(f[k], h5py.Dataset))
                return np.array(f[key])

    def make_training_data(self):
        for root, _, files in os.walk(self.label_dir):
            for fname in tqdm(files, desc=f"Scanning {root}"):
                if not fname.lower().endswith('.mat'):
                    continue
                if 'METADATA' in fname.upper():
                    continue

                path = os.path.join(root, fname)
                # only process GT_RAW or NOISY_RAW files
                if not any(tag in fname.upper() for tag in ('GT_RAW', 'NOISY_RAW')):
                    continue

                try:
                    raw = self._load_mat(path).astype(np.float32)

                    # normalize to [0,255]
                    raw -= raw.min()
                    if raw.max() > 0:
                        raw /= raw.max()
                    img = (raw * 255).astype(np.uint8)

                    # ensure 3-channel BGR
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[0] in (1,3) and img.ndim == 3:
                        # MATLAB sometimes stores channels first
                        img = np.transpose(img, (1,2,0))

                    # resize
                    img = cv2.resize(img, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_AREA)

                    self.training_data.append(img)

                except Exception as e:
                    print(f"Warning: could not process {fname}: {e}")

        np.save('training_data.npy', self.training_data)
        print(f"Saved {len(self.training_data)} samples to training_data.npy")

if REBUILD_DATA:
    denoiser = Denoising()
    denoiser.make_training_data()
