import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VarAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VarAutoEncoder, self).__init__():

        self.latent_dim = latent_dim
        self.fully_connected = nn.Flatten()
        self.average = nn.Linear(128*128, latent_dim)
        self.sigma = nn.Linear(128*128, latent_dim)

        self.encoder_layers = nn.Sequential(
            #batch_size x 3 x 256 x 256
            nn.Conv2d(3,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #batch_size x 64 x 256 x 256
            nn.Conv2d(64,128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #batch_size x 128 x 256 x 256
            nn.MaxPool2d(2, stride=2),
            #batch_size x 128 x 128 x 128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #batch_size x 512 x 128 x 128
            nn.LeakyReLU()
        )

        self.decoder_layers = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU,
            nn.Conv2d(256,128),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,3),
            nn.LeakyReLU()
        )

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        # encoder network
        x = self.encoder_layer(x) #should add residual blocks
        x = x.view(x.size(0), -1)

        #latent space
        mu = self.average(x)
        sigma = self.sigma(x)
        z = self.reparameterize(mu, sigma)
        nn.Linear(self.latent_dim, 128*128),
        nn.ReLU()
        #batch_size x 512 x 128 x 128

        x_hat = self.decode_layers(z)
        return x_hat, mu, sigma






# class Autoencoder(nn.Module):
#     def __init__(self, latent_dims,output_dims):
#         super(Autoencoder, self).__init__()
#         self.encoder = ResNet18(latent_dims)
#         self.decoder = Decoder(latent_dims,output_dims)

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z)

# class BasicBlock(nn.Module):
#   expansion = 1

#   def __init__(self, in_planes, planes, stride):
#     super(BasicBlock, self).__init__()
#     self.conv1 = nn.Conv2d(
#         in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                            stride =1, padding=1, bias=False)
#     self.bn2 = nn.BatchNorm2d(planes)

#     self.shortcut = nn.Sequential()
#     if stride != 1 or in_planes != self.expansion*planes:
#       self.shortcut = nn.Sequential(
#           nn.Conv2d(in_planes, self.expansion*planes,
#                     kernel_size=1, stride=stride, bias=False),
#           nn.BatchNorm2d(self.expansion*planes)
#       )
#   def forward(self, x):
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = self.bn2(self.conv2(out))
#     out += self.shortcut(x)
#     out = F.relu(out)
#     return out
  
# class ResNetEncoder(nn.Module):
#     def __init__(self, block, num_blocks, latent_dims):
#         super(ResNetEncoder, self).__init__()
#         self.in_planes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
#         # Additional fully connected layer to create the latent representation
#         x = torch.randn((1, 3, 256, 256))
#         self._initialize_fc(x, latent_dims)

#     def _initialize_fc(self, x, latent_dims):
#         with torch.no_grad():
#             x = self.forward_features(x)
#             num_features_before_fc = x.view(x.size(0), -1).shape[1]
        
#         self.fc = nn.Linear(num_features_before_fc, latent_dims)

#     def forward_features(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = F.avg_pool2d(x, 4)
#         mu = 
#         return x

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x



# def ResNet18(latent_dim):
#   return ResNetEncoder(BasicBlock, [2, 2, 2, 2],latent_dim)


# class Decoder(nn.Module):
#     def __init__(self, latent_dims, output_dims):
#         super(Decoder, self).__init__()
#         self.output_dims = output_dims
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, np.prod(output_dims))

#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         return z.view((-1, *self.output_dims))