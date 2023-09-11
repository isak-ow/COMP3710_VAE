import torch
import torch.nn as nn
import VAE_classes as VAE
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import gdown
import zipfile
import os

#importing data, ugly function
def get_dataset(folder_name: str, file_id: str='1ohzlvNEadhUTVkWtTz3oFn59TI8ahAPl'):
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
    ])
    dataset = datasets.ImageFolder(root=folder_name, transform=transform)
    return dataset

#defining the stochastic loss function using the input, estimate, average and the variance
def loss_func(x, x_hat, mu, sigma):
    result = nn.functional.binary_cross_entropy(x,x_hat,reduction='sum')
    result += (mu.pow(2)+sigma.pow(2)-torch.log(sigma)-0.5).sum()
    return result

#dimension of latent space should be 2d or 3d for plotting
latent_dim = 2
batch_size = 2


oasis = get_dataset('oasis_data')
print('dataset downloaded!')
dataloader = DataLoader(oasis, batch_size, shuffle = True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#defining model
model = VAE.Autoencoder(latent_dim,[256,256])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())

#training loop
num_epochs = 1  
for epoch in range(num_epochs):
    mu, sigma = 0,0
    for _, batch in enumerate(dataloader):
        x = batch[0]
        images = images.to(device)

        optimizer.zero_grad()
        x_hat, mu, sigma = model(images)
        loss = loss_func(x, x_hat, mu, sigma)
        loss.backward()
        optimizer.step()
    
    print('Epoch: {}, Loss: {:.5f}, mu: {:.5f}, sigma: {:.5f}'
          .format(epoch, loss, mu, sigma))
    break