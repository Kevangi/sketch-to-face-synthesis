"""
augmentation.py
---------------
On-the-fly augmentation for paired sketch-photo data.

Key constraint: sketch and photo MUST receive the exact same spatial
transforms (flip, rotate, crop) so they stay aligned. Only the photo
receives colour-based transforms (jitter, gamma) since sketches are
grayscale edge drawings.

Usage in dataset.py:
    from augmentation import PairedAugmentation

    augmentor = PairedAugmentation(img_size=256, augment=True)
    sketch_aug, photo_aug = augmentor(sketch_np, photo_np)
"""

import cv2
import random
import numpy as np


class PairedAugmentation:
    """
    Applies identical spatial transforms to both sketch and photo,
    and colour transforms to photo only.

    Args:
        img_size : expected input/output size (square)
        augment  : if False, only normalises to [-1,1] — no augmentation.
                   Set True for train split, False for test split.

    Input:
        sketch_np : uint8 numpy array (H, W, 3) — grayscale-valued RGB
        photo_np  : uint8 numpy array (H, W, 3) — RGB

    Output:
        sketch_np, photo_np : float32 numpy arrays (H, W, 3) in [-1, 1]
    """

    def __init__(self, img_size: int = 256, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment

    def __call__(self, sketch_np: np.ndarray, photo_np: np.ndarray):
        if self.augment:
            sketch_np, photo_np = self._apply_spatial(sketch_np, photo_np)
            photo_np            = self._apply_colour(photo_np)

        sketch_np = self._to_float(sketch_np)
        photo_np  = self._to_float(photo_np)

        return sketch_np, photo_np

    # ------------------------------------------------------------------
    # Spatial transforms — applied identically to both images
    # ------------------------------------------------------------------

    def _apply_spatial(self, sketch, photo):
        """
        All spatial ops use the same random params for sketch and photo
        so they remain geometrically aligned after augmentation.
        Order: flip -> rotate -> random crop
        """

        # 1. Random horizontal flip (p=0.5)
        if random.random() < 0.5:
            sketch = cv2.flip(sketch, 1)
            photo  = cv2.flip(photo,  1)

        # 2. Random rotation +-10 degrees (p=0.5)
        if random.random() < 0.5:
            angle  = random.uniform(-10, 10)
            sketch = self._rotate(sketch, angle)
            photo  = self._rotate(photo,  angle)

        # 3. Random crop then resize back to img_size (p=0.5)
        #    Crops 85-100% of the image — slight zoom effect
        if random.random() < 0.5:
            scale         = random.uniform(0.85, 1.0)
            sketch, photo = self._random_crop(sketch, photo, scale)

        return sketch, photo

    def _rotate(self, img, angle):
        h, w   = img.shape[:2]
        cx, cy = w // 2, h // 2
        M      = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _random_crop(self, sketch, photo, scale):
        h, w  = sketch.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Same random top-left corner for both images
        top  = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        sketch = sketch[top:top + new_h, left:left + new_w]
        photo  = photo [top:top + new_h, left:left + new_w]

        # Resize back to original img_size
        sketch = cv2.resize(sketch, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        photo  = cv2.resize(photo,  (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)

        return sketch, photo

    # ------------------------------------------------------------------
    # Colour transforms — photo only
    # ------------------------------------------------------------------

    def _apply_colour(self, photo):
        """
        Colour ops applied to photo only.
        Sketches are grayscale so colour transforms would distort
        edge structure and are meaningless on a single-channel signal.
        """

        # 1. Brightness jitter +-20% (p=0.5)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            photo  = self._adjust_brightness(photo, factor)

        # 2. Contrast jitter +-20% (p=0.5)
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            photo  = self._adjust_contrast(photo, factor)

        # 3. Saturation jitter +-15% (p=0.5)
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            photo  = self._adjust_saturation(photo, factor)

        # 4. Random gamma correction — simulates exposure variation (p=0.3)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            photo = self._adjust_gamma(photo, gamma)

        return photo

    def _adjust_brightness(self, img, factor):
        img = img.astype(np.float32) * factor
        return np.clip(img, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, img, factor):
        mean = img.mean()
        img  = (img.astype(np.float32) - mean) * factor + mean
        return np.clip(img, 0, 255).astype(np.uint8)

    def _adjust_saturation(self, img, factor):
        hsv          = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _adjust_gamma(self, img, gamma):
        inv_gamma = 1.0 / gamma
        table     = np.array([
            (i / 255.0) ** inv_gamma * 255
            for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(img, table)

    # ------------------------------------------------------------------
    # Normalise to [-1, 1] — always applied (train and test)
    # ------------------------------------------------------------------

    def _to_float(self, img):
        """Convert uint8 [0, 255] -> float32 [-1, 1]."""
        return img.astype(np.float32) / 127.5 - 1.0

    def __repr__(self):
        return (
            f'PairedAugmentation('
            f'img_size={self.img_size}, '
            f'augment={self.augment})'
        )


# ------------------------------------------------------------------
# Smoke test — run directly: python augmentation.py
# ------------------------------------------------------------------

if __name__ == '__main__':

    print('Running augmentation smoke test...')

    H, W   = 256, 256
    sketch = np.ones((H, W, 3), dtype=np.uint8) * 220
    photo  = np.zeros((H, W, 3), dtype=np.uint8)
    photo[:, :, 0] = 180
    photo[:, :, 1] = 140
    photo[:, :, 2] = 120

    # Draw circle so alignment is visible
    cv2.circle(sketch, (128, 128), 60, (50,  50,  50), 2)
    cv2.circle(photo,  (128, 128), 60, (100, 80,  60), 2)

    aug = PairedAugmentation(img_size=256, augment=True)

    for i in range(5):
        s_aug, p_aug = aug(sketch.copy(), photo.copy())
        assert s_aug.shape == (256, 256, 3),             f'Sketch shape wrong: {s_aug.shape}'
        assert p_aug.shape == (256, 256, 3),             f'Photo shape wrong: {p_aug.shape}'
        assert s_aug.dtype == np.float32,                f'Sketch dtype wrong: {s_aug.dtype}'
        assert p_aug.dtype == np.float32,                f'Photo dtype wrong: {p_aug.dtype}'
        assert s_aug.min() >= -1.01 and s_aug.max() <= 1.01, f'Sketch out of range: [{s_aug.min():.2f}, {s_aug.max():.2f}]'
        assert p_aug.min() >= -1.01 and p_aug.max() <= 1.01, f'Photo out of range: [{p_aug.min():.2f}, {p_aug.max():.2f}]'
        print(f'  Pass {i+1}: '
              f'sketch {s_aug.shape} [{s_aug.min():.2f}, {s_aug.max():.2f}]  '
              f'photo  {p_aug.shape} [{p_aug.min():.2f}, {p_aug.max():.2f}]')

    # Test with augment=False (test split — only normalise)
    aug_off      = PairedAugmentation(img_size=256, augment=False)
    s_off, p_off = aug_off(sketch.copy(), photo.copy())
    assert s_off.shape == p_off.shape == (256, 256, 3)
    assert s_off.dtype == np.float32
    print(f'  No-augment pass: OK')

    print('\nAll smoke tests passed.')
    print(aug)
    print(aug_off)
