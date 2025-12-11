# GanS

Generative Adversarial Networks Implementation using PyTorch

## 💫 Overview

This project implements a comprehensive Generative Adversarial Network (GAN) using PyTorch for generating synthetic handwritten digit images similar to the MNIST dataset. It demonstrates the core principles of adversarial learning, including a Generator network that creates synthetic images and a Discriminator network that learns to distinguish real from fake images.

**Project Objectives:**
- Understand GAN architecture and adversarial training dynamics
- Implement a fully functional GAN from scratch using PyTorch
- Generate high-quality synthetic handwritten digits
- Visualize learning progression through generated sample grids
- Explore hyperparameter tuning and training optimization

## 🚀 Key Features

### Architecture Components
- **Generator**: Fully connected neural network that transforms random noise (latent vector) into synthetic images
- **Discriminator**: Binary classifier that distinguishes real MNIST images from generated fake images
- **Loss Functions**: Binary Cross-Entropy (BCE) for adversarial training
- **Optimization**: Adam optimizer for both Generator and Discriminator

### Training & Visualization
- **Adversarial Training**: Alternating optimization of Generator and Discriminator
- **Sample Grid Generation**: Visual output grids generated at each epoch to track learning
- **Training Metrics**: Loss tracking for both networks to monitor convergence
- **Hyperparameter Configuration**: Customizable learning rates, batch sizes, and latent dimensions

### Data Processing
- **MNIST Dataset**: Classic 28x28 pixel handwritten digit images (0-9)
- **Data Normalization**: Pixel values normalized to [-1, 1] range
- **Data Augmentation**: Optional transforms for improved training robustness
- **Batch Processing**: Efficient DataLoader with configurable batch sizes

## 📄 Project Structure

```
GanS/
├── README.md                          # Project documentation
├── GaNS_application_using_MNIST.py    # Main training script
├── requirements.txt                    # Python dependencies
├── generated_samples/                 # Output directory for generated images
└── checkpoints/                       # Model weights checkpoints
```

## 💪 Neural Network Architecture

### Generator Network

**Input**: Latent vector z ~ N(0, 1) of dimension (batch_size, latent_dim)

**Architecture**:
```
Latent Vector (100-dim)
    ↓
Linear Layer (100 -> 256) + BatchNorm + ReLU
    ↓
Linear Layer (256 -> 512) + BatchNorm + ReLU
    ↓
Linear Layer (512 -> 1024) + BatchNorm + ReLU
    ↓
Linear Layer (1024 -> 784) + Tanh
    ↓
Reshaped to (1, 28, 28)
```

**Output**: Synthetic image of shape (batch_size, 1, 28, 28) with values in [-1, 1]

### Discriminator Network

**Input**: Image of shape (batch_size, 1, 28, 28)

**Architecture**:
```
Flattened Image (784)
    ↓
Linear Layer (784 -> 512) + LeakyReLU(0.2)
    ↓
Dropout(0.3)
    ↓
Linear Layer (512 -> 256) + LeakyReLU(0.2)
    ↓
Dropout(0.3)
    ↓
Linear Layer (256 -> 1) + Sigmoid
    ↓
Binary Classification Output [0, 1]
```

**Output**: Probability score indicating if image is real (close to 1) or fake (close to 0)

## 📚 Training Process

### Adversarial Learning Loop

**Per Iteration:**

1. **Discriminator Update**:
   - Train on real MNIST images (label = 1)
   - Train on generated fake images (label = 0)
   - Calculate combined loss and backpropagate

2. **Generator Update**:
   - Generate fake images from random noise
   - Pass through Discriminator (trying to fool it)
   - Minimize loss (making Discriminator output close to 1)

3. **Loss Functions**:
   - Discriminator: L_D = -log(D(x)) - log(1 - D(G(z)))
   - Generator: L_G = -log(D(G(z)))

### Training Dynamics

- **Convergence**: Networks reach equilibrium where Generator creates realistic images
- **Mode Collapse**: Risk that Generator learns to produce limited variety (handled with proper hyperparameters)
- **Spectral Normalization**: Optional technique to stabilize Discriminator training

## 🙋 Installation & Setup

**Requirements**:
- Python 3.7+
- PyTorch 1.9+
- torchvision
- NumPy
- Matplotlib
- Pillow

**Installation Steps**:

```bash
# Clone repository
git clone https://github.com/ritvikvr/GanS.git
cd GanS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR install manually:
pip install torch torchvision numpy matplotlib pillow
```

**GPU Setup** (Optional but recommended):
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Usage

### Basic Training

```bash
python GaNS_application_using_MNIST.py
```

This will:
1. Download MNIST dataset automatically
2. Initialize Generator and Discriminator networks
3. Train for specified epochs
4. Save generated sample grids to `./generated_samples/`
5. Save model checkpoints to `./checkpoints/`

### Custom Configuration

Edit hyperparameters in the script:

```python
# Model Configuration
latent_dim = 100          # Dimension of latent vector
image_size = 28           # MNIST image size
batch_size = 64           # Training batch size
num_epochs = 100          # Number of training epochs

# Optimizer Configuration
learning_rate_g = 0.0002  # Generator learning rate
learning_rate_d = 0.0002  # Discriminator learning rate
beta1 = 0.5               # Adam beta1 parameter
beta2 = 0.999             # Adam beta2 parameter

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Advanced Usage

```python
import torch
from GaNS_application_using_MNIST import Generator, train_gan

# Load pre-trained generator
generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load('checkpoints/generator_final.pth'))
generator.eval()

# Generate new samples
z = torch.randn(64, 100).to(device)
generated_images = generator(z)
```

## 📊 Training Metrics & Results

### Expected Performance

**Training Progress**:
- Epoch 1-10: Generator produces noisy patterns
- Epoch 10-30: Recognizable digit-like shapes emerge
- Epoch 30-50: Clear digit structures with some variations
- Epoch 50+: High-quality synthetic MNIST-like digits

**Loss Curves**:
- Discriminator Loss: Stabilizes around 0.5-1.0
- Generator Loss: Decreases as Generator improves
- Both should reach equilibrium (not one winning completely)

**Quality Metrics**:
- **Inception Score**: Measures diversity and quality of generated images
- **Fréchet Distance**: Compares generated vs real image distributions
- **Visual Inspection**: Sample grids saved every epoch for qualitative evaluation

## 퇟f️ Hyperparameter Tuning Guide

### Learning Rate Adjustment
```python
learning_rate_g = 0.0001  # Lower: more stable but slower convergence
learning_rate_d = 0.0004  # Discriminator usually needs higher LR
```

### Architecture Tuning
```python
# Increase complexity for better image quality
hidden_dims = [256, 512, 1024, 2048]  # More layers & units
latent_dim = 200          # Higher dimensional latent space
```

### Training Stability
```python
# Prevent mode collapse
batch_size = 128          # Larger batches
beta1 = 0.5               # Lower momentum in Adam
label_smoothing = 0.9     # Smooth labels (0 -> 0.1, 1 -> 0.9)
```

## 📂 Common Issues & Solutions

### Mode Collapse
**Problem**: Generator produces only a few digit types
**Solutions**:
- Increase latent dimension
- Use label smoothing
- Adjust discriminator-to-generator loss ratio
- Use spectral normalization

### Unstable Training
**Problem**: Losses oscillate wildly
**Solutions**:
- Lower learning rates
- Increase batch size
- Use batch normalization in generator
- Add gradient clipping

### Poor Image Quality
**Problem**: Generated images are blurry
**Solutions**:
- Increase training epochs
- Deeper network architectures
- Better hyperparameter tuning
- Add regularization techniques

## 🙋 Contributing

Contribution areas:
- [ ] Implement conditional GANs (c-GAN) for controlled digit generation
- [ ] Add Wasserstein GAN loss for improved training stability
- [ ] Implement progressive training strategy
- [ ] Add evaluation metrics (Inception Score, Fréchet Distance)
- [ ] Create web interface for real-time generation
- [ ] Multi-GPU training support
- [ ] Experiment with different architectures (ResNet, DenseNet blocks)

## 📂 License & Citation

MIT License - See LICENSE file

If you use this project, please cite:

```bibtex
@misc{ritvik2024gans,
  author = {Ritvik Verma},
  title = {Generative Adversarial Networks: PyTorch Implementation},
  year = {2024},
  url = {https://github.com/ritvikvr/GanS},
  note = {MNIST digit generation using adversarial learning}
}
```

## 🙋 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **GAN Paper**: Ian Goodfellow et al., "Generative Adversarial Networks" (2014)
- **PyTorch Framework**: Meta AI Research
- **Community**: Deep Learning research community for foundational work

## 👤 Author

**Ritvik Verma** (@ritvikvr)
Computer Science Engineering Student | AI/Data Science Specialization
GitHub: https://github.com/ritvikvr

Interested in Generative AI, Deep Learning, Computer Vision, GANs

---

*Last Updated: December 2024*
Feel free to star ⭐ if you find this project helpful!
