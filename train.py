import os
import time
import glob
import torch
from copy import deepcopy
from losses import *

def train(data, true_values, model,
          project='AE',
          imputation='ZINBGAE',
          no_cuda=False,
          epochs=3000,
          lr=0.001,
          weight_decay=0,
          patience=200,
          alpha=0.5,
          beta=0.5,
          gamma=0.5,
          fastmode=False,
          verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossFunc = torch.nn.MSELoss()
    zinb_loss = ZINBLoss()
    data=data.to(device)

    def train_wrapper(epoch):
        model.train()
        optimizer.zero_grad()
        
        if project == 'AE' and imputation == 'GAE':
            
            output_mlp, pred = model(data)
            loss_proj = sparse_mse_loss(output_mlp,true_values)
            loss_imp = lossFunc(pred, true_values)
            
        if project == 'AE' and imputation == 'ZINBGAE':
            
            output_mlp, pred, _mean, _disp, _pi = model(data)
            loss_proj = sparse_mse_loss(output_mlp, true_values)
            loss_imp = zinb_loss(true_values, pred, _mean, _disp, _pi, beta=gamma)
            
        if project == 'ZINBAE' and imputation == 'GAE':
            
            output_mlp, neck1, neck2, neck3, pred = model(data)
            loss_proj = zinb_loss(true_values, output_mlp, neck1, neck2, neck3, beta=beta)
            loss_imp = lossFunc(pred, true_values)
            
        if project == 'ZINBAE' and imputation == 'ZINBGAE':
            
            output_mlp, neck1, neck2, neck3, pred, _mean, _disp, _pi = model(data)
            loss_proj = zinb_loss(true_values, output_mlp, neck1, neck2, neck3, beta=beta)
            loss_imp = zinb_loss(true_values, pred, _mean, _disp, _pi, beta=gamma)

            
        total_loss=alpha * loss_proj + (1 - alpha) * loss_imp

        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 and verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(total_loss.data.item()))

        return total_loss.data.item()

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = float('inf')
    best_epoch = 0
    best_model_state_dict = None  # store the best model's state_dict

    for epoch in range(epochs):
        loss_values.append(train_wrapper(epoch))

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            best_model_state_dict = deepcopy(model.state_dict())  # Save the best model's state_dict
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    # Restore best model
    model.load_state_dict(best_model_state_dict)
    #torch.save(model.state_dict(), 'best_model.pkl')  # Save the best model