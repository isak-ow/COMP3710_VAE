import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from VAE_script import VAE
from utils import get_dataset, plot_input_output
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = get_dataset('vae_mri')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load your VAE model
latent_dim = 2  # Assuming your latent space is 2D
model = VAE(latent_dim)
model.load_state_dict(torch.load('vae_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Assuming 'dataset' is your dataset and 'vae_model' is your trained VAE
latent_vectors = []
for batch in dataloader:
    data = batch[0].to(device)  # Assuming you're using GPU
    with torch.no_grad():
        mu, _ = model.encode(data)
        latent_vectors.append(mu.cpu().numpy())

latent_vectors = np.vstack(latent_vectors)

# Visualize the 2D latent space
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1])
plt.xlabel("Dim1")
plt.ylabel("Dim2")

#Save the fig
plt.savefig('latent_space.png')