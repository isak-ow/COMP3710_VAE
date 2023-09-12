import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
from VAE_script import VAE
import os

def plot_input_output(random_image, reconstructed_image):
    # Assuming 'reconstructed_image' is in range [0, 1]
    reconstructed_image = reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy()  # Adjust dimensions for plotting

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(random_image.squeeze().permute(1, 2, 0).detach().numpy())  # Assuming random_image is in [0, 1]
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image)
    plt.axis('off')

    plt.tight_layout()
    #Return the plot
    return plt

def plot_generated_image(generated_image):
    # Assuming 'generated_image' is in range [0, 1]
    generated_image = generated_image.squeeze().permute(1, 2, 0).detach().numpy()  # Adjust dimensions for plotting

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))

    # Generated Image
    plt.subplot(1, 1, 1)
    plt.title("Generated Image")
    plt.imshow(generated_image)
    plt.axis('off')

    plt.tight_layout()
    #Return the plot
    return plt

model = VAE(2)
model.load_state_dict(torch.load('vae_model.pth'))  
model.eval() 

num_samples = 10  # Number of images you want to generate
random_latent_vectors = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    generated_images = model.decode(random_latent_vectors)
    
# Create a folder to save the generated images
os.makedirs("generated_images", exist_ok=True)

# Assuming `generated_images` is a torch tensor of generated images
for i in range(num_samples):
    plt = plot_generated_image(generated_images[i])
    plt.savefig(f"generated_images/generated_image_{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()