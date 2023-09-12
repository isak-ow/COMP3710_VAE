import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import gdown
import zipfile
import os
import wandb

wandb.init(project="OSASIS_VAE_rangpur", name="VAE")

def get_dataset(folder_name: str, file_id: str='1azN6agg-u-0NnNkDgr0EO5escKTRA3s5'):
    #Check if the folder name exists
    if not os.path.exists(folder_name):
        output = 'oasis_data.zip'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(output)
    else:
        print('Dataset is already downloaded')
    transform = tt.Compose([
        tt.Resize((256, 256)),
        tt.ToTensor(),
        #tt.Normalize([0.0013, 0.0013, 0.0013],[0.0022, 0.0022, 0.0022])
    ])
    dataset = datasets.ImageFolder(root=folder_name, transform=transform)
    return dataset

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            #nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 16 * 16)
        
        self.decoder_conv = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure outputs are in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(z.size(0), 256, 16, 16)
        x = self.decoder_conv(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
oasis = get_dataset('keras_png_slices_data')

#dimension of latent space should be 2d or 3d for plotting
print('getting dataset')
oasis = get_dataset('keras_png_slices_data')
print('dataset downloaded!')
image, label = oasis[55]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

latent_dim = 2
batch_size = 32
model = VAE(latent_dim)
dataloader = DataLoader(oasis, batch_size, shuffle = True)

lr = 0.001
num_epochs = 25
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device('cuda')

# Training loop # Set this to the number of epochs you want to train for
model.to(device)
model.train()

for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (x,_) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        print(mu.shape)
        print(logvar.shape)
        
        #Loss computation
        loss = ((x - x_hat)**2).sum()
        loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #loss = loss_function(x_hat,x,mu,logvar)
        
        #Compute gradients and step the weights
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader.dataset):.4f}")
    wandb.log({'loss': train_loss/len(dataloader.dataset)})
    
print("Training complete.")
torch.save(model.state_dict(), 'vae_model.pth')
