# CIFAR Autoencoder for Image Restoration 🧠🖼️

This project implements a convolutional autoencoder trained on the CIFAR-10 dataset. The autoencoder learns to reconstruct original images from damaged (noisy) inputs, making it useful for tasks like denoising or minor image restoration.

---

## 🔧 Features

- Trains an autoencoder using PyTorch on CIFAR-10 dataset
- Adds Gaussian noise to simulate image damage
- Learns to reconstruct the original image
- Visualizes original, noisy, and restored images
- Easily extendable to different noise types or datasets

---

## 🗂️ Project Structure

```

cifar\_autoencoder/
├── data/                   # CIFAR-10 data gets downloaded here
├── models/
│   └── autoencoder.py      # Autoencoder architecture
├── utils/
│   └── noise.py            # Noise-adding utility function
├── train.py                # Model training script
├── test.py                 # Testing & visualization
├── config.py               # Hyperparameters and constants
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

````

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/LukaBabunadze/cifar_autoencoder.git
cd cifar_autoencoder
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This script will:

* Download CIFAR-10
* Add Gaussian noise to images
* Train the autoencoder
* Save the trained model as `autoencoder.pth`

### 4. Test and visualize reconstruction

```bash
python test.py
```

You’ll see:

* The original image
* The noisy (damaged) version
* The restored (reconstructed) image

---

## 🧠 Model Architecture

A simple convolutional autoencoder:

* **Encoder:** 3 Conv layers (stride 2) to compress image
* **Decoder:** 3 ConvTranspose layers to reconstruct image

All activations use **ReLU** except for the final layer which uses **Sigmoid**.

---

## 🧪 Noise Function

The `utils/noise.py` file adds Gaussian noise:

```python
noisy = images + noise_factor * torch.randn_like(images)
```

You can tune `noise_factor` to make restoration harder or easier.

---

## 📈 Example Output

| Original                  | Noisy                  | Reconstructed                  |
| ------------------------- | ---------------------- | ------------------------------ |
| ![](samples/original.png) | ![](samples/noisy.png) | ![](samples/reconstructed.png) |

---

## ⚙️ Configurable Parameters

Edit `config.py` to change:

```python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = 'cuda' or 'cpu'
```

---
