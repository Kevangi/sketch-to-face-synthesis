import cv2
import random
import numpy as np


class PairedAugmentation:
    """
    Args:
        img_size : square image size (default 256)
        augment  : True  -> spatial + colour + normalize  (train split)
                   False -> normalize only                 (test split)

    Input:
        sketch_np : uint8 numpy (H, W, 3) — grayscale-valued RGB, range [0, 255]
        photo_np  : uint8 numpy (H, W, 3) — RGB, range [0, 255]

    Output:
        sketch_np, photo_np : float32 numpy (H, W, 3), range [-1, 1]
    """

    def __init__(self, img_size: int = 256, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment

    def __call__(self, sketch_np: np.ndarray, photo_np: np.ndarray):
        if self.augment:
            sketch_np, photo_np = self._spatial(sketch_np, photo_np)
            photo_np            = self._colour(photo_np)

        # Normalize always — both train and test
        sketch_np = self._normalize(sketch_np)
        photo_np  = self._normalize(photo_np)

        return sketch_np, photo_np

    # Spatial — identical params for both images
    def _spatial(self, sketch, photo):

        # 1. Horizontal flip (p=0.5)
        if random.random() < 0.5:
            sketch = cv2.flip(sketch, 1)
            photo  = cv2.flip(photo,  1)

        # 2. Rotation +-10 degrees (p=0.5)
        if random.random() < 0.5:
            angle  = random.uniform(-10, 10)
            sketch = self._rotate(sketch, angle)
            photo  = self._rotate(photo,  angle)

        # 3. Random crop + resize back (p=0.5)
        if random.random() < 0.5:
            scale         = random.uniform(0.85, 1.0)
            sketch, photo = self._crop(sketch, photo, scale)

        return sketch, photo

    def _rotate(self, img, angle):
        h, w   = img.shape[:2]
        M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT_101)

    def _crop(self, sketch, photo, scale):
        h, w  = sketch.shape[:2]
        new_h = int(h * scale)
        new_w = int(w * scale)
        top   = random.randint(0, h - new_h)
        left  = random.randint(0, w - new_w)

        sketch = sketch[top:top + new_h, left:left + new_w]
        photo  = photo [top:top + new_h, left:left + new_w]

        sketch = cv2.resize(sketch, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        photo  = cv2.resize(photo,  (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)

        return sketch, photo

    # Colour — photo only
    def _colour(self, photo):

        # Brightness +-20% (p=0.5)
        if random.random() < 0.5:
            f     = random.uniform(0.8, 1.2)
            photo = np.clip(photo.astype(np.float32) * f, 0, 255).astype(np.uint8)

        # Contrast +-20% (p=0.5)
        if random.random() < 0.5:
            f     = random.uniform(0.8, 1.2)
            mean  = photo.mean()
            photo = np.clip((photo.astype(np.float32) - mean) * f + mean, 0, 255).astype(np.uint8)

        # Saturation +-15% (p=0.5)
        if random.random() < 0.5:
            f              = random.uniform(0.85, 1.15)
            hsv            = cv2.cvtColor(photo, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1]   = np.clip(hsv[:, :, 1] * f, 0, 255)
            photo          = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gamma correction (p=0.3)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                              for i in range(256)]).astype(np.uint8)
            photo = cv2.LUT(photo, table)

        return photo

    # Normalize — always applied
    def _normalize(self, img):
        """uint8 [0, 255] -> float32 [-1, 1]"""
        return img.astype(np.float32) / 127.5 - 1.0

    def __repr__(self):
        return f'PairedAugmentation(img_size={self.img_size}, augment={self.augment})'


# Smoke test — python augmentation.py
if __name__ == '__main__':
    print('Smoke test...')

    sketch = np.ones((256, 256, 3), dtype=np.uint8) * 220
    photo  = np.zeros((256, 256, 3), dtype=np.uint8)
    photo[:, :, 0] = 180
    photo[:, :, 1] = 140
    photo[:, :, 2] = 120
    cv2.circle(sketch, (128, 128), 60, (50, 50, 50), 2)
    cv2.circle(photo,  (128, 128), 60, (100, 80, 60), 2)

    for augment in [True, False]:
        aug = PairedAugmentation(img_size=256, augment=augment)
        s, p = aug(sketch.copy(), photo.copy())
        assert s.shape == p.shape == (256, 256, 3)
        assert s.dtype == p.dtype == np.float32
        assert s.min() >= -1.01 and s.max() <= 1.01
        assert p.min() >= -1.01 and p.max() <= 1.01
        print(f'  augment={augment}: sketch [{s.min():.2f}, {s.max():.2f}]  photo [{p.min():.2f}, {p.max():.2f}]  OK')

    print('All tests passed.')
