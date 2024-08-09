import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential, BatchNorm1d, Dropout, Linear
import torch_geometric
from torch_geometric.nn import DenseGCNConv, DenseGraphConv, GCNConv, GraphNorm
from torch.utils.data import DataLoader
import torch_scatter
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from modules import *
import numpy as np
import os


class IntegratedModel(nn.Module):
    def __init__(self, x, size_factors, project='AE', imputation='GAE', k=5, distance='euclidean', z_dim=128, y_dim=256,
                 pro_dim1=128, pro_dim2=1024, dropout=0.0, use_raw=True):
        super(IntegratedModel, self).__init__()
        self.mlp = MLP(x, projection_dim=z_dim, dropout=dropout)
        self.vae = VAE(x, pro_dim1=z_dim, pro_dim2=y_dim)
        self.edgeSamplingGumbel = EdgeSamplingGumbel(k=k, distance=distance)
        self.gae = GAE(x)
        self.vgae = VGAE(x, pro_dim1, pro_dim2)
        self.size_factors=size_factors
        self.project = project
        self.imputation = imputation
        self.use_raw=use_raw

    def forward(self,x):
        
        x_t=x.clone().t()
        
        if self.project == 'AE' and self.imputation == 'GAE':
            # Classic autoencoder
            #_output = []
            x_imp,bottleneck_output = self.mlp(x)
            x_, edge_index, edge_weights, adj=self.edgeSamplingGumbel(bottleneck_output)
            if self.use_raw:
                #print(edge_index.shape)
               # GraphAutoencoder
                x_rec = self.gae(x, edge_index, self.size_factors)
            else:
                x_rec = self.gae(x_imp, edge_index, self.size_factors)
            return x_imp, x_rec
        
        if self.project == 'AE' and self.imputation == 'VGAE':
            x_imp,bottleneck_output = self.mlp(x)
            x_, edge_index, edge_weights, adj=self.edgeSamplingGumbel(bottleneck_output)
            if self.use_raw:
                # VGAE  
                #print('x shape:',x.shape)
                #print('adj:',adj.shape)
                x_rec, z_mean, z_dropout, z_dispersion = self.vgae(x, adj, x_t, self.size_factors)
            else:
                x_rec, z_mean, z_dropout, z_dispersion = self.vgae(x_imp, adj, x_t, self.size_factors)
            return x_imp, x_rec, z_mean, z_dropout, z_dispersion
        
        if self.project == 'VAE' and self.imputation == 'GAE':
            x_imp, hidden1, hidden2, hidden3, bottleneck_output = self.vae(x, x_t, size_factors=1.0)
            x_, edge_index, edge_weights, adj=self.edgeSamplingGumbel(bottleneck_output)
            if self.use_raw:
                #print(edge_index.shape)
               # GraphAutoencoder
                x_rec = self.gae(x, edge_index, self.size_factors)
            else:
                x_rec = self.gae(x_imp, edge_index, self.size_factors)
            return x_imp, hidden1, hidden2, hidden3, x_rec
        
        if self.project == 'VAE' and self.imputation == 'VGAE':
            x_imp, hidden1, hidden2, hidden3, bottleneck_output = self.vae(x, x_t, size_factors=1.0)
            x_, edge_index, edge_weights, adj=self.edgeSamplingGumbel(bottleneck_output)
            #print(edge_weights.shape)
            #print(edge_index.shape)
            if self.use_raw:
                # VGAE  
                #print('x shape:',x.shape)
                #print('adj:',adj.shape)
                x_rec, z_mean, z_dropout, z_dispersion = self.vgae(x, adj, x_t, self.size_factors)
            else:
                x_rec, z_mean, z_dropout, z_dispersion = self.vgae(x_imp, adj, x_t, self.size_factors)
            return x_imp, hidden1, hidden2, hidden3, x_rec, z_mean, z_dropout, z_dispersion