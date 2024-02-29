import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track, BinaryTrack
from tqdm import tqdm
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time as time
from IPython import display
from torch.utils.data import Dataset, DataLoader

class GeneratorBlock3d(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)

class GeneratorBlock2d(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose2d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm2d(out_dim)
    
    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)
    
class Generator_paper2_all_s(nn.Module):

    def __init__(self, latent_dim, n_tracks, beat_resolution, n_pitches):

        super().__init__()
        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        
        if beat_resolution == 4:
            self.k1, self.k2 = 2, 4
        elif beat_resolution == 12:
            self.k1, self.k2 = 4, 6

        self.octaves = n_pitches // 12

        self.g_bar = nn.Sequential(
            GeneratorBlock2d(self.latent_dim, 512, (self.k1, 1), (self.k1, 1)),
            GeneratorBlock2d(512, 256, (2, 1), (2, 1)),
            GeneratorBlock2d(256, 128, (2, 1), (2, 1)),
            GeneratorBlock2d(128, 128, (2, 1), (2, 1)),
            GeneratorBlock2d(128, 64, (self.k2, 1), (self.k2, 1)),
            GeneratorBlock2d(64, 64, (1, self.octaves), (1, self.octaves)),
            
            nn.ConvTranspose2d(64, self.n_tracks, (1,12), stride = (1,12)),
            nn.BatchNorm2d((self.n_tracks)),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        y = self.g_bar(x)
        return y
    
class Generator_paper2_cond(nn.Module):

    def __init__(self, latent_dim, n_tracks, beat_resolution, n_pitches):

        super().__init__()
        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        self.ln = False ### DA SETTARE CORRETTAMENTE
        
        if beat_resolution == 4:
            self.k1, self.k2 = 2, 4
        elif beat_resolution == 12:
            self.k1, self.k2 = 4, 6

        self.octaves = n_pitches // 12
        
        
        self.e1 =  DiscriminatorBlock2d(self.n_tracks, 64, (1, 12), (1, 12), self.ln) # [-1, 64, 64, 7]
        self.e2 =  DiscriminatorBlock2d(64, 64, (1, self.octaves), (1, self.octaves), self.ln) # [-1, 64, 64, 1]
        self.e3 =  DiscriminatorBlock2d(64, 64, (self.k2, 1), (self.k2, 1), self.ln) # [-1, 64, 16, 1]
        self.e4 =  DiscriminatorBlock2d(64, 64, (2, 1), (2, 1), self.ln) # [-1, 64, 8, 1]
        self.e5 =  DiscriminatorBlock2d(64, 64, (2, 1), (2, 1), self.ln) # [-1, 64, 4, 1]
        #self.e6 =  DiscriminatorBlock2d(64, 64, (2, 1), (2, 1), self.ln) # [-1, 64, 2, 1]
        #self.e7 =  DiscriminatorBlock2d(64, 1, (2, 1), (2, 1), self.ln) # [-1, 1, 1, 1]
        
        self.g1 = GeneratorBlock2d(self.latent_dim, 512, (self.k1, 1), (self.k1, 1)) # [-1, 512, 2, 1]
        self.g2 = GeneratorBlock2d(512, 256, (2, 1), (2, 1)) # [-1, 256, 4, 1]
        self.g3 = GeneratorBlock2d(256+64, 128, (2, 1), (2, 1)) # [-1, 128, 8, 1]
        self.g4 = GeneratorBlock2d(128+64, 128, (2, 1), (2, 1)) # [-1, 128, 16, 1]
        self.g5 = GeneratorBlock2d(128+64, 64, (self.k2, 1), (self.k2, 1)) # [-1, 64, 64, 1]
        self.g6 = GeneratorBlock2d(64+64, 64, (1, self.octaves), (1, self.octaves)) # [-1, 64, 64, 7]
        self.g7 = nn.ConvTranspose2d(64+64, self.n_tracks, (1,12), stride = (1,12)) # [-1, 1, 64, 84]
        
        self.bn2d = nn.BatchNorm2d((self.n_tracks))
        self.tanh = nn.Tanh()
        
    def forward(self, x, x_prev):
        
        h1_prev = self.e1(x_prev)
        h2_prev = self.e2(h1_prev)
        h3_prev = self.e3(h2_prev)
        h4_prev = self.e4(h3_prev)
        h5_prev = self.e5(h4_prev)
        
        x = x.view(-1, self.latent_dim, 1, 1)
        h1 = self.g1(x)
        h2 = self.g2(h1)
        h2 = torch.cat((h2, h5_prev), 1)
        h3 = self.g3(h2)
        h3 = torch.cat((h3, h4_prev), 1)
        h4 = self.g4(h3)
        h4 = torch.cat((h4, h3_prev), 1)
        h5 = self.g5(h4)
        h5 = torch.cat((h5, h2_prev), 1)
        h6 = self.g6(h5)
        h6 = torch.cat((h6, h1_prev), 1)
        h7 = self.g7(h6)
        y = self.bn2d(h7)
        y = self.tanh(y)

        return y

class Generator_mine(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self, latent_dim, n_tracks, beat_resolution, n_pitches):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.n_pitches = n_pitches
        
        if beat_resolution == 4:
            self.k1, self.k2 = 2, 2
        elif beat_resolution == 12:
            self.k1, self.k2 = 4, 3
        
        self.octaves = n_pitches // 12

        self.transconv0 = GeneratorBlock3d(self.latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneratorBlock3d(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneratorBlock3d(128, 64, (1, 1, self.octaves), (1, 1, self.octaves))
        self.transconv3 = torch.nn.ModuleList([
            GeneratorBlock3d(64, 32, (1, self.k1, 1), (1, self.k1, 1))
            for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        self.transconv4 = torch.nn.ModuleList([
            GeneratorBlock3d(32, 16, (1, self.k2, 1), (1, self.k2, 1))
            for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        self.transconv5 = torch.nn.ModuleList([
            GeneratorBlock3d(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        
        ### IMPORTANT: THERE IS NOT TANH ACTIVATION FUNCTION
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = [transconv(x) for transconv in self.transconv3] # list of n_tracks different processing of the x
        x = [transconv(x_) for x_, transconv in zip(x, self.transconv4)] # each x_ inside the list has a different processing, producing another list
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1) # each x_ inside the list has a different processing, than all is concatenated
        
        x = x.view(-1, self.n_tracks, 4*4*self.beat_resolution, self.n_pitches)
        
        #x = self.tanh(x) ### optional, not working
        return x

    
class DiscriminatorBlock3d(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, add_layernorm = False):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
        self.add_layernorm = add_layernorm
    
    def forward(self, x):
        x = self.conv(x)
        if self.add_layernorm:
            x = self.layernorm(x)
            
        return torch.nn.functional.leaky_relu(x)
    
class DiscriminatorBlock2d(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, add_layernorm = False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
        self.add_layernorm = add_layernorm
    
    def forward(self, x):
        x = self.conv(x)
        if self.add_layernorm:
            x = self.layernorm(x)
            
        return torch.nn.functional.leaky_relu(x)
    
class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class Discriminator_paper2_all(nn.Module):

    def __init__(self, n_tracks, beat_resolution, n_pitches, add_layernorm, add_dropout):

        super().__init__()

        if beat_resolution == 4:
            self.k1, self.k2 = 2, 2
        elif beat_resolution == 12:
            self.k1, self.k2 = 4, 3
        
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.n_pitches = n_pitches
        self.octaves = n_pitches // 12
        self.ln = add_layernorm
        self.drop = add_dropout

        self.conv = nn.Sequential(
            DiscriminatorBlock3d(self.n_tracks, 64, (4, 1, 1), (4, 1, 1), self.ln),
            DiscriminatorBlock3d(64, 128, (1, 1, 12), (1, 1, 12), self.ln),
            DiscriminatorBlock3d(128, 128, (1, 1, self.octaves), (1, 1, self.octaves), self.ln),
            DiscriminatorBlock3d(128, 128, (1, 2, 1), (1, 2, 1), self.ln),
            DiscriminatorBlock3d(128, 128, (1, 2, 1), (1, 2, 1), self.ln),
            DiscriminatorBlock3d(128, 128, (1, self.k1, 1), (1, self.k1, 1), self.ln),
            DiscriminatorBlock3d(128, 512, (1, self.k2, 1), (1, self.k2, 1), self.ln)
        )

        self.fc_net1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU()
        )
        
        if self.drop:
            self.dropout = nn.Dropout(p = 0.2)
        
        self.fc_net2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, self.n_tracks, 4, 4*self.beat_resolution, self.n_pitches)
        y = self.conv(x)
        y = y.view(-1, 512)

        if self.drop:
            y = self.dropout(y)
            
        y = self.fc_net1(y)

        if self.drop:
            y = self.dropout(y)

        y = self.fc_net2(y)

        return y
    
class Discriminator_mine(torch.nn.Module):
    """A convolutional neural network (CNN) based discriminator. The
    discriminator takes as input either a real sample (in the training data) or
    a fake sample (generated by the generator) and outputs a scalar indicating
    its authentity.
    """
    def __init__(self, n_tracks, beat_resolution, n_pitches, add_layernorm, add_dropout):
        super().__init__()

        if beat_resolution == 4:
            self.k1, self.k2 = 2, 2
        elif beat_resolution == 12:
            self.k1, self.k2 = 4, 3
        
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.n_pitches = n_pitches
        self.octaves = n_pitches // 12
        self.ln = add_layernorm
        self.drop = add_dropout

        self.conv0 = torch.nn.ModuleList([
            DiscriminatorBlock3d(1, 16, (1, 1, 12), (1, 1, 12), self.ln) for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorBlock3d(16, 16, (1, self.k2, 1), (1, self.k2, 1), self.ln) for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        self.conv2 = torch.nn.ModuleList([
            DiscriminatorBlock3d(16, 16, (1, self.k1, 1), (1, self.k1, 1), self.ln) for _ in range(self.n_tracks)
        ]) # list of n_tracks blocks
        self.conv3 = DiscriminatorBlock3d(16 * self.n_tracks, 64, (1, 1, self.octaves), (1, 1, self.octaves), self.ln)
        self.conv4 = DiscriminatorBlock3d(64, 128, (1, 4, 1), (1, 4, 1), self.ln)
        self.conv5 = DiscriminatorBlock3d(128, 256, (4, 1, 1), (4, 1, 1), self.ln)
        self.dense = torch.nn.Linear(256, 1)

        if self.drop:
            self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        
        x = x.view(-1, self.n_tracks, 4, 4*self.beat_resolution, self.n_pitches)

        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)] # list of n_tracks different processing of the elements of x
        x = [conv(x_) for x_, conv in zip(x, self.conv1)] # list of n_tracks different processing of the elements of x
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv2)], 1) # list of n_tracks different processing of the elements of x, than all is concatenated
        x = self.conv3(x)          
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256)

        if self.drop:
            x = self.dropout(x)

        x = self.dense(x)

        return x