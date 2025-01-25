# scE2EGAE
Enhancing Single-cell RNA-Seq Data Analysis through an End-to-End Cell-Graph-Learnable Graph Autoencoder with Differentiable Edge Sampling  

[The preprint manuscript at Research Square](https://doi.org/10.21203/rs.3.rs-5279794/v1)

## Introduction about code scripts and hyperparameters 

### Codes

   distances.py: functions used for distances metrix calculation

   modules.py: codes of all the modules we used. The modules.py mainly contains the following classes:
   
               a) EdgeSamplingGumbel(nn.Module): For edge sampling 
               
               b) GAE(torch.nn.Module): The graph autoencoder for single-cell RNAseq data denoising 
               
               c) ZINBAE(Module): The DCA module for projecting single-cell features (counts) into a lower-dimensional space.

   losses.py: codes of the loss functions

   imputation_model.py: the intergrated model comprises from the modules

   train.py: the pytorch train wrapper codes

   Main.ipynb: codes to train the model and some simple visualization
   
### Hyperparameters: please follow this table when initializing the model

| Datasets | Num_Epochs | Patience | K | Distance_Measure | AE_Dim | GAE_Dim | Dropout_GAE | LR    | Alpha  | Beta | MSE_V2 | 
|----------|------------|----------|---|------------------|--------|---------|-------------|-------|--------|------|--------|
| Klein    | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.0005 | 1    | False  | 
| Zeisel   | 700        | 30       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | True   |
| Romanov  | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.0005 | 1    | False  |
| ITC      | 500        | 20       | 1 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | False  |
| Chu      | 2000       | 20       | 1 | Euclidean        | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  |
| ILC      | 4000       | 10       | 1 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  |
| Tirosh   | 800        | 10       | 1 | Hyperbolic       | 128    | 128     | 0           | 0.003 | 0.001  | 1    | False  |
| AD       | 1500       | 20       | 1 | Hyperbolic       | 128    | 64      | 0           | 0.003 | 0.001  | 1    | False  |


## How to run the code

### 1. First, we create an Anaconda environment:
   
   conda create -n pytorch python=3.9
   
   conda activate pytorch
   
### 2. Then, install the needed packages:
   
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   
   conda install pyg -c pyg
   
   pip install anndata==0.8.0
   
   pip install notebook
   
   pip install h5py==3.7.0
   
   pip install loompy==3.0.7
   
   pip install matplotlib==3.7.1
   
   pip install numpy==1.23.4
   
   pip install pandas==1.5.1
   
   pip install scanpy==1.9.3
   
   pip install scikit-learn==1.1.3
   
   pip install scipy==1.9.3

### 3. Clone the repo to your PC:

    git clone https://github.com/PPDPQ/scE2EGAE.git

### 4. If you have raw data, you can use the notebooks in the /data_prep folder. If you do not have them, due to GitHub's file upload size limit of 25MB, we have uploaded a processed Zeisel dataset to the "/dataset" folder. You can try running our code using this dataset.

### 5. Run the "Main.ipynb" notebook cell by cell.




