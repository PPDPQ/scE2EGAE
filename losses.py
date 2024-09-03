import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import DataLoader  
import torch_geometric  
import torch_scatter  
from torch_geometric.data import Data  
from torch.nn import MSELoss 
import numpy as np

def _nan2zero(x):  
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)  
  
def _nan2inf(x):  
    return torch.where(torch.isnan(x), torch.full_like(x, float('inf')), x)  
  
def _nelem(x):  
    nelem = torch.sum(~torch.isnan(x)).float()  
    return torch.where(nelem == 0., torch.tensor(1.).to(x.device), nelem).to(x.dtype)  
  
def _reduce_mean(x):  
    nelem = _nelem(x)  
    x = _nan2zero(x)  
    return torch.sum(x) / nelem 

def mse_loss_v2(y_true, y_pred):  
    y_true = torch.log(y_true + 1)  
    y_pred = torch.log(y_pred + 1)  
    ret = (y_pred - y_true) ** 2  
    return _reduce_mean(ret)
  
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
        recon_loss = mse_loss_v2(x_true, x_recon)
        
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