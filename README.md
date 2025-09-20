# GanS

Generative Adversarial Networks – PyTorch Implementation
Overview
This project demonstrates a simple Generative Adversarial Network (GAN) using PyTorch for generating handwritten digit images similar to MNIST data. The implementation provides end-to-end code for model design, training, and result visualization.
Features
	•	Minimal GAN architecture with fully connected Generator and Discriminator.
	•	Uses MNIST dataset for digital image generation.
	•	Includes model hyperparameter setup, data loading, and image normalization.
	•	Automated saving of generated sample grids for every epoch.
	•	Generates visual output for each epoch to track learning quality progression.
 
Usage
Requirements
	•	Python with PyTorch, torchvision, glob, and IPython.display libraries.
	•	CUDA-enabled GPU recommended for faster training.
 
Running the Code
	1.	Set up the project directory and install dependencies.
	2.	Place the provided code in a Python script (e.g., `gan_mnist.py`).
	3.	Run the script. The script will:
	•	Download MNIST, apply data transforms, and create a DataLoader.
	•	Define simple Generator and Discriminator classes using PyTorch’s `nn.Module`.
	•	Train both networks using the adversarial strategy described in the GAN framework.
	•	Periodically save generated images to `/content/generated` (customizable via `save_path`).
	4.	At the end of training, open the saved `.png` images to visualize results across epochs.

 Output
	•	Each epoch produces a grid of generated samples, letting users observe learning improvement.
	•	The last generated image offers the best synthetic samples after the final epoch.
 
Code Highlights
	•	Generator: Maps random noise vectors to fake images through two linear layers and Tanh activation.
	•	Discriminator: Classifies 784-dimensional inputs as real or fake using LeakyReLU and Sigmoid layers.
	•	Loss: Binary cross-entropy for both Generator and Discriminator.
	•	Optimizers: Adam for both networks; learning rate = 0.0002.
	•	Normalization: Input images are normalized to .
	•	Visualization: Uses torchvision utilities and IPython to display generated image grids for qualitative analysis.
