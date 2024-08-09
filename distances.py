import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    #dist = torch.cdist(x,x)**2
    dist = torch.cdist(x,x,p=2)
    return dist, x

#Cosine distance
def pairwise_cosine_distances(x, dim=-1):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=dim)
    similarity = torch.matmul(x_norm, x_norm.t())
    distance = 1 - similarity
    return distance, x

#Manhattan distance
def pairwise_manhattan_distances(x, dim=-1):
    dist = torch.cdist(x, x, p=1)
    return dist, x

# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x**2).sum(dim,keepdim=True)
    x_norm = (x_norm.sqrt()-1).relu() + 1 
    x = x/(x_norm*(1+1e-2))
    x_norm = (x**2).sum(dim,keepdim=True)
    
    pq = torch.cdist(x,x)**2
    dist = torch.arccosh(1e-6+1+2*pq/((1-x_norm)*(1-x_norm.transpose(-1,-2))))**2
    return dist, x