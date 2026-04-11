# sketch-to-face-synthesis

A deep learning project that converts hand-drawn face sketches into realistic face photos using Pix2Pix GAN with AdaFace identity loss.

---

## What This Project Does

Given a sketch of a face, the model generates a realistic-looking photo of that person. It is trained on the CUHK Face Sketch dataset which has 188 paired sketch-photo images.

---

## Dataset

- **CUHK Face Sketch Database** — 188 sketch-photo pairs
- Split into **150 train** and **38 test** images
- All images resized to **256×256**
- Download from https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs

---

| File | What it does |
|------|-------------|
| `load_dataset.ipynb` | Downloads the CUHK dataset from Kaggle and verifies all 188 pairs |
| `preprocessing_spilt.ipynb` | Resizes images to 256×256, converts sketches to grayscale, splits into train/test |
| `augmentation.py` | Applies random flips, rotations, colour changes during training |
| `dataset.py` | Loads sketch-photo pairs and feeds them into the model during training |
| `pix2pix.py` | The model — U-Net generator and PatchGAN discriminator |
| `train_pix2pix.ipynb` | Trains the model for 200 epochs on Google Colab |
| `evaluate_pix2pix.ipynb` | Tests the trained model and computes SSIM, FID, and face identity scores |

--- 

## Getting Started

### Step 1 — Clone the Repository


```bash
git clone https://github.com/your-username/Sketch2Face.git
cd Sketch2Face
```

### Step 2 — Install Dependencies

```bash
pip install torch torchvision opencv-python-headless tqdm piq torchmetrics pytorch-msssim scipy opencv-contrib-python-headless
```

### Step 3 — Set Up Kaggle API

You need a Kaggle account to download the dataset.

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
2. This downloads a `kaggle.json` file to your computer
3. Keep it ready — you will upload it in the next step

### Step 4 — Download the Dataset

Open and run `load_dataset.ipynb`.
When prompted, upload your `kaggle.json` file.
This will download and extract the CUHK dataset into `dataset/`.

### Step 5 — Preprocess and Split the Data

Open and run `preprocessing_spilt.ipynb`.
This will:
- Resize all images to 256×256
- Convert sketches to grayscale
- Split the data into 150 train and 38 test pairs

### Step 6 — Train the Model

Open and run `train_pix2pix.ipynb`.

> A GPU is recommended for training. Training runs for 200 epochs and saves checkpoints to `output_pix2pix_adaface/checkpoints/`.

### Step 7 — Evaluate the Model

Open and run `evaluate_pix2pix.ipynb`.
This loads the trained model and computes SSIM, FID, and SFace identity scores on the test set. Results are saved to `results/`.

---

## Model

- **Generator**: U-Net with 8 encoder/decoder blocks and skip connections
- **Discriminator**: 70×70 PatchGAN
- **Loss**: Adversarial loss + L1 loss + Feature Matching loss + AdaFace identity loss

---

## Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 200 |
| Batch size | 4 |
| Image size | 256×256 |
| Learning rate | 2e-4 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| LR schedule | Flat for first 100 epochs, then linear decay to 0 |
| Generator filters (ngf) | 64 |
| Discriminator filters (ndf) | 64 |
| Weight init | Normal (mean=0, std=0.02) |
| Random seed | 42 |

### Loss Weights

| Loss | Weight |
|------|--------|
| L1 pixel loss | 100.0 |
| VGG perceptual loss | 10.0 |
| Feature matching loss | 10.0 |
| AdaFace identity loss | 10.0 |

### Other Settings

| Setting | Value |
|---------|-------|
| Save checkpoint every | 20 epochs |
| Save sample images every | 10 epochs |
| Replay buffer size | 50 |
| Dataloader workers | 2 |

---

## Results

Best checkpoint: **Epoch 160**

FID                      : 74.0887   (lower is better)
SSIM                     : 0.7199    (higher is better)
MS-SSIM                  : 0.8378    (higher is better)
FSIM                     : 0.8175    (higher is better)
VIF                      : 0.2360    (higher is better)
SR-SIM                   : 0.8707    (higher is better)
IS                       : 1.3632    (higher is better)
Rank-1                   : 92.11%    (higher is better)
VR @ FAR 0.1%            : 65.79%    (higher is better)
VR @ FAR 1%              : 86.84%    (higher is better)

## References

- [Pix2Pix paper](https://arxiv.org/abs/1611.07004)
- [AdaFace paper](https://arxiv.org/abs/2204.00964)