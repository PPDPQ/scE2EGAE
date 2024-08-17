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
  
# Defines the Zero-Inflated Negative Binomial (ZINB) Loss for ZINB-based autoencoder, 
# modified based on https://github.com/DHUDBlab/scDSC/blob/master/layers.py
# Because the original target of the author is clustering, a MSE loss is added for the imputation task
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()
    
    def reconstruction_loss(self, x_true, x_recon):
        recon_loss = MSELoss()(x_recon, x_true)
        
        return recon_loss

    def forward(self, x_true, x_recon, mean, disp, pi, beta=0.1, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        #scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x_true+1.0) - torch.lgamma(x_true+disp+eps)
        t2 = (disp+x_true) * torch.log(1.0 + (mean/(disp+eps))) + (x_true * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        zinb_loss = torch.where(torch.le(x_true, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            zinb_loss += ridge
        zinb_loss = torch.mean(zinb_loss)
        
        recon_loss = self.reconstruction_loss(x_recon, x_true)
        
        result = beta * zinb_loss + (1 - beta) * recon_loss
        return result