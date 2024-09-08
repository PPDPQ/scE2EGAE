# scE2EGAE
End-to-end dropouts imputation on single-cell RNA-Seq data with differentiable cell-cell graph learning graph-autoencoder

## Hyperparameters 
| Datasets | Num_Epochs | Patience | K | Distance_Measure | AE_Dim | GAE_Dim | Dropout_GAE | LR    | Alpha  | Beta | MSE_V2 | 
|----------|------------|----------|---|------------------|--------|---------|-------------|-------|--------|------|--------|
| Klein    | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.0005 | 1    | False  | 
| Zeisel   | 700        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | True   |
| Romanov  | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.0005 | 1    | False  |
| ITC      | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | False  |
| Chu      | 2000       | 20       | 3 | Euclidean        | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  |
| ILC      | 4000       | 10       | 1 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  |
| Tirosh   | 800        | 10       | 1 | Hyperbolic       | 128    | 128     | 0           | 0.003 | 0.001  | 1    | False  |
| AD       | 1500       | 20       | 1 | Hyperbolic       | 128    | 64      | 0           | 0.003 | 0.001  | 1    | False  |

## Umap comparisons

![image](https://github.com/PPDPQ/E2EGraphImpute/blob/main/umap00_all.jpg)

