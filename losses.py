import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  
import torch_geometric  
import torch_scatter  
from torch_geometric.data import Data  
from torch.nn import MSELoss 
import numpy as np
  
# Defines a custom loss function that combines MSE loss with a sparsity penalty  
def sparse_mse_loss(preds, labels, sparsity_weight=0.2):  
    mse_loss = F.mse_loss(preds, labels)  
    sparsity_loss = torch.mean(torch.abs(preds))  # Sparsity penalty  
    total_loss = sparsity_weight * sparsity_loss + (1 - sparsity_weight) * mse_loss 
    return total_loss  
  
# Defines the Zero-Inflated Negative Binomial (ZINB) Loss for VGAE, modified based on https://github.com/inoue0426/scVGAE 
def ZINBLoss(y_true, y_pred, theta, pi, eps=1e-10):  
    # Negative Binomial Loss component  
    nb_terms = (  
        -torch.lgamma(y_true + theta)  
        + torch.lgamma(y_true + 1)  
        + torch.lgamma(theta)  
        - theta * torch.log(theta + eps)  
        + theta * torch.log(theta + y_pred + eps)  
        - y_true * torch.log(y_pred + theta + eps)  
        + y_true * torch.log(y_pred + eps)  
    )  
  
    # Zero-Inflation component  
    zero_inflated = torch.log(pi + (1 - pi) * torch.pow(1 + y_pred / theta, -theta))  
  
    # Combine both components, applying zero-inflation only for zero counts  
    result = -torch.sum(  
        zero_inflated * (y_true < eps).float()  
        + (1 - (y_true < eps).float()) * nb_terms  
    )  
  
    return torch.round(result, decimals=3)  
  
# Computes a combined loss: ZINB Loss + MSE Loss  
def compute_loss(x_original, x_recon, z_mean, z_dropout, z_dispersion, gamma):  
    # Compute ZINB Loss  
    zinb_loss = ZINBLoss(x_original, z_mean, z_dispersion, z_dropout)  
  
    # Compute MSE Loss  
    mse_loss = MSELoss()(x_recon, x_original)  
  
    # Normalize ZINB Loss by the number of elements in x_recon and combine with MSE Loss (The modified part)  
    total_loss = gamma * (zinb_loss / (x_recon.shape[0] * x_recon.shape[1])) + (1 - gamma) * mse_loss
    #total_loss = gamma * zinb_loss + (1 - gamma) * mse_loss
  
    return total_loss