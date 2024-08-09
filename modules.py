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
import numpy as np
import os
from distances import pairwise_euclidean_distances, pairwise_cosine_distances, pairwise_manhattan_distances, pairwise_poincare_distances

# Module for generating edges based on a given distance metric and Gumbel sampling
class EdgeSamplingGumbel(nn.Module):
    def __init__(self, k=5, distance='hyperbolic'):
        super(EdgeSamplingGumbel, self).__init__()

        self.k = k
        self.eps=1e-8
        # Set distance function and temperature parameter based on input
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
            self.temperature = nn.Parameter(torch.tensor(4).float(), requires_grad = True)
        if distance == 'cosine':
            self.distance = pairwise_cosine_distances
            self.temperature = nn.Parameter(torch.tensor(4.).float(), requires_grad = True)
        if distance == 'manhattan':
            self.distance = pairwise_manhattan_distances
            self.temperature = nn.Parameter(torch.tensor(1.).float(), requires_grad = True)
        if distance == 'hyperbolic':
            self.distance = pairwise_poincare_distances
            self.temperature = nn.Parameter(torch.tensor(5.).float(), requires_grad = True)
        
    def forward(self, x): 
        
        if self.training:
            
            dist, _x = self.distance(x)
           
            edge_index, edge_weights = self.gumbel_top_k(dist) 
                
        else:
            with torch.no_grad():
                dist, _x = self.distance(x)

                edge_index, edge_weights = self.gumbel_top_k(dist)
        
       
        row = edge_index.t()[:, 0]  
        col = edge_index.t()[:, 1]  
        num_nodes = int(max(row.max(), col.max())) + 1
        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        #print(adj)

        return x, edge_index, edge_weights, adj
    

    def gumbel_top_k(self, distance_mx):
        num_nodes = distance_mx.shape[0]
        distance_mx = distance_mx * torch.exp(torch.clamp(self.temperature,-5,5))
        #distance_mx = torch.exp(-self.temperature * distance_mx) + self.eps
        
        q = torch.rand_like(distance_mx)
        lq = (distance_mx-torch.log(-torch.log(q)))
        #lq=torch.nn.functional.gumbel_softmax(distance_mx)
        
        logprobs, indices = torch.topk(-lq,self.k)  
    
        rows = torch.arange(num_nodes).view(num_nodes,1).to(distance_mx.device).repeat(1,self.k)
        edges = torch.stack((indices.view(-1),rows.view(-1)),-2)
        
        return edges, logprobs.view(-1)

# Flatten layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
# Reshape layer    
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)
    
#Autoencoder for projection
class MLP(nn.Module): 
    def __init__(self, x, projection_dim=32, final_activation=False, dropout=0.2):
        super(MLP, self).__init__()
        self.x = x
        self.projection_dim = projection_dim
        layers_size = [self.x.shape[-1], projection_dim, self.x.shape[-1]]
        layers = []

        for li in range(1, len(layers_size)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li-1], layers_size[li]))
            if li == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
            
        self.MLP = nn.Sequential(*layers)
        self.bottleneck_output = None
        
    def forward(self, e=None):
        x_imp = self.x
        self.bottleneck_output = self.MLP[:3](x_imp)  # Assuming the bottleneck layer is the middle linear layer
        x_imp = self.MLP(x_imp)
        return x_imp,self.bottleneck_output

# Autoencoder using ZINB loss
class VAE(Module):
    def __init__(self, x, pro_dim1=128, pro_dim2=1024):
        super(VAE, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

        # Encoder with 2 Linear layers
        self.dense1 = Linear(x.shape[1], pro_dim1)
        self.batchn1 = BatchNorm1d(pro_dim1)
        self.dense2_mean = Linear(pro_dim1, x.shape[1])
        self.dense2_dropout = Linear(pro_dim1, x.shape[1])
        self.dense2_dispersion = Linear(pro_dim1, x.shape[1])

        # Decoder with 2 Linear layers
        self.fc1 = Linear(x.shape[1], pro_dim2)
        self.bn2 = BatchNorm1d(pro_dim2)
        self.fc2 = Linear(pro_dim2, x.shape[1])

        self.batch_norm1 = BatchNorm1d(x.shape[1])
        self.batch_norm2 = BatchNorm1d(x.shape[0])

    def encode(self, x):
        x = self.batchn1(self.dense1(x)) #being used to generate edges
        x = F.relu(x) 
        x = self.dropout1(x)

        z_mean = torch.exp(self.dense2_mean(x))
        z_dropout = torch.sigmoid(self.dense2_dropout(x))
        z_dispersion = torch.exp(self.dense2_dispersion(x))
        return z_mean, z_dropout, z_dispersion

    def decode(self, z):
        neck = F.relu(self.bn2(self.fc1(z)))
        #z = F.relu(neck)
        z = self.dropout2(neck)
        return neck, F.relu(self.fc2(z))

    def forward(self, x, x_t, size_factors): 
        z_mean, z_dropout, z_dispersion = self.encode(x)
        neck, x_recon = self.decode(z_mean)
        #x_recon = size_factors*(x_recon + self.batch_norm1(x) + self.batch_norm2(x_t).T)
        x_recon = size_factors*(x_recon)
        x_recon = F.relu(x_recon)
        return x_recon, z_mean, z_dropout, z_dispersion, neck

# Graph Autoencoder (GAE)
class GAE(torch.nn.Module):
    #def __init__(self,input_dim=None,hidden_dim=64,impute_dim=None):
    def __init__(self,x):
        super(GAE, self).__init__()
        #self.num_features=x.shape[-1]
        self.conv1 = GCNConv(x.shape[-1], int(x.shape[-1]/64))
        self.encode_ln = torch.nn.LayerNorm(int(x.shape[-1]/64))
        self.conv2 = GCNConv(int(x.shape[-1]/64), x.shape[-1])
        
        #self.conv1 = GCNConv(input_dim, hidden_dim)
        #self.conv2 = GCNConv(hidden_dim, impute_dim)

    def forward(self, x, edge_index,size_factors):
        x = F.relu(self.encode_ln(self.conv1(x, edge_index)))
        x = F.dropout(x,p=0.0, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x=x*size_factors
        return x
    
# Graph Autoencoder using ZINB loss
class VGAE(Module):
    def __init__(self, x, pro_dim1=128, pro_dim2=1024):
        super(VGAE, self).__init__()

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

        # Encoder with 2 gcn layers
        self.gcn1 = GCNConv(x.shape[1], pro_dim1)
        self.gn1 = GraphNorm(pro_dim1)
        self.gcn2_mean = GCNConv(pro_dim1, x.shape[1])
        self.gcn2_dropout = GCNConv(pro_dim1, x.shape[1])
        self.gcn2_dispersion = GCNConv(pro_dim1, x.shape[1])

        # Decoder with 2 Linear layers
        self.fc1 = Linear(x.shape[1], pro_dim2)
        self.bn2 = BatchNorm1d(pro_dim2)
        self.fc2 = Linear(pro_dim2, x.shape[1])

        self.batch_norm1 = BatchNorm1d(x.shape[1])
        self.batch_norm2 = BatchNorm1d(x.shape[0])

    def encode(self, x, adj):
        x = F.relu(self.gn1(self.gcn1(x, adj)))
        x = self.dropout1(x)

        z_mean = torch.exp(self.gcn2_mean(x, adj.t()))
        z_dropout = torch.sigmoid(self.gcn2_dropout(x, adj.t()))
        z_dispersion = torch.exp(self.gcn2_dispersion(x, adj.t()))
        return z_mean, z_dropout, z_dispersion

    def decode(self, z):
        z = F.relu(self.bn2(self.fc1(z)))
        z = self.dropout2(z)
        return F.relu(self.fc2(z))

    def forward(self, x, adj,x_t, size_factors):  
        z_mean, z_dropout, z_dispersion = self.encode(x, adj.t())
        #x_recon = size_factors*(self.decode(z_mean) + self.batch_norm1(x) + self.batch_norm2(x_t).T)
        x_recon = size_factors*(self.decode(z_mean))
        x_recon = F.relu(x_recon)
        return x_recon, z_mean, z_dropout, z_dispersion

