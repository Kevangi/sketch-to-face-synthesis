import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation import PairedAugmentation


class CUHKDataset(Dataset):
    """
    Args:
        split_dir : path to train/ or test/ inside data/split/
        augment   : True  -> augmentation + normalize  (train)
                    False -> normalize only             (test)
    """

    VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, split_dir: str, augment: bool = False):
        super().__init__()

        self.sketch_dir = os.path.join(split_dir, 'sketches')
        self.photo_dir  = os.path.join(split_dir, 'photos')

        if not os.path.exists(self.sketch_dir):
            raise FileNotFoundError(f'Sketches folder not found: {self.sketch_dir}')
        if not os.path.exists(self.photo_dir):
            raise FileNotFoundError(f'Photos folder not found: {self.photo_dir}')

        self.augment   = augment
        self.augmentor = PairedAugmentation(img_size=256, augment=augment)
        self.pairs     = self._scan_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f'No matched pairs found in {split_dir}')

        print(f'CUHKDataset | {os.path.basename(split_dir)} | '
              f'{len(self.pairs)} pairs | augment={augment}')

    # Scan folders and match by filename stem

    def _scan_pairs(self):
        def scan(folder):
            return {
                os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(folder, '*'))
                if os.path.splitext(p)[1].lower() in self.VALID_EXTS
            }

        sketch_map = scan(self.sketch_dir)
        photo_map  = scan(self.photo_dir)

        common     = sorted(set(sketch_map) & set(photo_map))

        unmatched_s = set(sketch_map) - set(photo_map)
        unmatched_p = set(photo_map)  - set(sketch_map)
        if unmatched_s:
            print(f'  Warning: {len(unmatched_s)} sketch(es) with no matching photo — skipped')
        if unmatched_p:
            print(f'  Warning: {len(unmatched_p)} photo(s) with no matching sketch — skipped')

        return [(sketch_map[sid], photo_map[sid]) for sid in common]

    # Dataset interface
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.pairs[idx]

        # Load uint8 RGB from disk
        sketch_np = self._load(sketch_path)   # (H, W, 3) uint8
        photo_np  = self._load(photo_path)    # (H, W, 3) uint8

        # Augment + normalize -> float32 [-1, 1]
        sketch_np, photo_np = self.augmentor(sketch_np, photo_np)

        # (H, W, C) -> (C, H, W) torch tensor
        return {
            'sketch'      : self._to_tensor(sketch_np),
            'photo'       : self._to_tensor(photo_np),
            'sketch_path' : sketch_path,
            'photo_path'  : photo_path,
        }

    # Helpers

    def _load(self, path: str) -> np.ndarray:
        """Load PNG, return uint8 RGB (H, W, 3)."""
        img = cv2.imread(path)
        if img is None:
            raise IOError(f'Cannot read: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """float32 numpy (H,W,C) -> float32 torch (C,H,W)."""
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """[-1,1] -> [0,1] for visualisation."""
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    def __repr__(self):
        return f'CUHKDataset(pairs={len(self.pairs)}, augment={self.augment})'
