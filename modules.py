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
        #self.k = nn.Parameter(torch.tensor(k).float(), requires_grad = True)
        self.eps=1e-8
        # Set distance function and temperature parameter based on input
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
            self.temperature = nn.Parameter(torch.tensor(1.).float(), requires_grad = True)
        if distance == 'cosine':
            self.distance = pairwise_cosine_distances
            self.temperature = nn.Parameter(torch.tensor(1.).float(), requires_grad = True)
        if distance == 'manhattan':
            self.distance = pairwise_manhattan_distances
            self.temperature = nn.Parameter(torch.tensor(1.).float(), requires_grad = True)
        if distance == 'hyperbolic':
            self.distance = pairwise_poincare_distances
            self.temperature = nn.Parameter(torch.tensor(1.).float(), requires_grad = True)
        
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
    
    '''
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
    '''
    
    def gumbel_top_k(self, distance_mx): 
        #_k = torch.round(torch.clamp(self.k, 0, 10)).int().detach()
        num_nodes = distance_mx.shape[0]  
        temperature = torch.clamp(self.temperature, 0, 5)  
        logits = -distance_mx * torch.exp(temperature)  
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  
        logits_with_gumbel = (logits + gumbel_noise) / temperature  
        y_soft = torch.nn.functional.softmax(logits_with_gumbel, dim=1)  

        # use straight-through estimator  
        hard_indices = torch.argmax(y_soft, dim=1)  
        y_hard = torch.zeros_like(y_soft).scatter_(1, hard_indices.unsqueeze(1), 1.0)  

        # use hard samples in the forward pass and soft samples for the backward pass  
        y = (y_hard - y_soft).detach() + y_soft  

        # obtain top-k factors  
        logprobs, indices = torch.topk(y_soft, self.k, dim=1) 
        #logprobs, indices = torch.topk(y, _k, dim=1) 

        rows = torch.arange(num_nodes).view(num_nodes, 1).to(distance_mx.device).repeat(1, self.k)  
        #rows = torch.arange(num_nodes).view(num_nodes, 1).to(distance_mx.device).repeat(1, _k)  
        edges = torch.stack((rows.view(-1), indices.view(-1)), -2)  

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

# Graph Autoencoder (GAE)
class GAE(torch.nn.Module):
    #def __init__(self,input_dim=None,hidden_dim=64,impute_dim=None):
    def __init__(self, x, hidden_dim=128, dropout=0.3):
        super(GAE, self).__init__()
        #self.num_features=x.shape[-1]
        self.dropout = dropout
        self.conv1 = GCNConv(x.shape[-1], hidden_dim)
        self.encode_ln = torch.nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, x.shape[-1])
        
        #self.conv1 = GCNConv(input_dim, hidden_dim)
        #self.conv2 = GCNConv(hidden_dim, impute_dim)

    def forward(self, x, edge_index, size_factors=1.0):
        x = F.relu(self.encode_ln(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x=x*size_factors
        return x
    
# Autoencoder and Graph Autoencoder using ZINB loss
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

# Autoencoder 
'''
class ZINBAE(Module):  
    def __init__(self, x, pro_dim1=40, pro_dim2=400):  
        super(ZINBAE, self).__init__()  
  
        # Encoder layers  
        self.enc1 = Linear(x.shape[1], pro_dim1)  
        self.bn1 = BatchNorm1d(pro_dim1)  
        self.dr1 = nn.Dropout(0.2)
        
        self.mean_act = MeanAct()  
        self.disp_act = DispAct()  
        self.pi_act = nn.Sigmoid()  
        self.dense2_mean = Linear(pro_dim1, x.shape[1])  
        self.dense2_disp = Linear(pro_dim1, x.shape[1])  
        self.dense2_pi = Linear(pro_dim1, x.shape[1]) 
        
        self.dec1 = Linear(x.shape[1], pro_dim2)  
        self.bn2 = BatchNorm1d(pro_dim2)  
        self.dr2 = nn.Dropout(0.4)  
        self.dec2 = Linear(pro_dim2, x.shape[1])    
  
    def forward(self, x, size_factors=1.0):  
        # Encoder  
        x = self.bn1(self.enc1(x))  
        x = F.relu(x)  
        neck = self.dr1(x)
  
        # Decoder 
        _mean = self.mean_act(self.dense2_mean(neck))   
        _disp = self.disp_act(self.dense2_disp(neck)) 
        _pi = self.pi_act(self.dense2_pi(neck)) 
        
        latent = F.relu(self.bn2(self.dec1(_mean)))  
        latent = self.dr2(latent)  
        x_recon = F.relu(self.dec2(latent))  
        x_recon = size_factors * x_recon   
  
        return x_recon, _mean, _disp, _pi, neck
'''
class ZINBAE(Module):  
    def __init__(self, x, hidden_dim=40):  
        super(ZINBAE, self).__init__()  
  
        # Encoder layers  
        self.enc1 = Linear(x.shape[1], hidden_dim)  
        self.bn1 = BatchNorm1d(hidden_dim)  
        self.dr1 = nn.Dropout(0.3)
        
        #Decoder layers
        self.mean_act = MeanAct()  
        self.disp_act = DispAct()  
        self.pi_act = nn.Sigmoid()  
        self.dense2_mean = Linear(hidden_dim, x.shape[1])  
        self.dense2_disp = Linear(hidden_dim, x.shape[1])  
        self.dense2_pi = Linear(hidden_dim, x.shape[1]) 
        
    def forward(self, x):  
        # Encoder  
        x = self.bn1(self.enc1(x))  
        x = F.relu(x)  
        neck = self.dr1(x)
  
        # Decoder 
        _mean = self.mean_act(self.dense2_mean(neck))   
        _disp = self.disp_act(self.dense2_disp(neck)) 
        _pi = self.pi_act(self.dense2_pi(neck)) 
  
        return _mean, _disp, _pi, neck


#Graph autoencoder
class ZINBGAE(Module):  
    def __init__(self, x, hidden_dim=40):  
        super(ZINBGAE, self).__init__()  
          
        # Graph encoder  
        self.gcn_share = GCNConv(x.shape[1], hidden_dim)  
        self.gn1 = GraphNorm(hidden_dim)  
        self.dr1 = nn.Dropout(0.3)  
          
        # Graph decoder  
        self.mean_act = MeanAct()  
        self.disp_act = DispAct()  
        self.pi_act = nn.Sigmoid()  
        self.gcn_mean = GCNConv(hidden_dim, x.shape[1])  
        self.gcn_disp = GCNConv(hidden_dim, x.shape[1])  
        self.gcn_pi = GCNConv(hidden_dim, x.shape[1])  
  
    def forward(self, x, adj, size_factors=1.0):  
        # Encoding  
        hidden = F.relu(self.gn1(self.gcn_share(x, adj)))  
        hidden = self.dr1(hidden)  
          
        # Decoding  
        _mean = self.mean_act(self.gcn_mean(hidden, adj))  
        _disp = self.disp_act(self.gcn_disp(hidden, adj))  
        _pi = self.pi_act(self.gcn_pi(hidden, adj))  
          
        return _mean, _disp, _pi
