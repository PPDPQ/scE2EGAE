# E2EGraphImpute
End-to-end dropouts imputation on single-cell RNA-Seq data with differentiable cell-cell graph learning graph-autoencoder

## Hyperparameters 
| Datasets | Num_Epochs | Patience | K | Distance_Measure | AE_Dim | GAE_Dim | Dropout_GAE | LR    | Alpha  | Beta | MSE_V2 | Use_Umap |
|----------|------------|----------|---|------------------|--------|---------|-------------|-------|--------|------|--------|----------|
| Klein    | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.0005 | 1    | False  | True     |
| Zeisel   | 700        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | True   | False    |
| Romanov  | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.0005 | 1    | False  | False    |
| ITC      | 500        | 20       | 3 | Hyperbolic       | 128    | 2000    | 0.1         | 0.003 | 0.001  | 1    | False  | False    |
| Chu      | 2000       | 20       | 3 | Euclidean        | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  | True     |
| ILC      | 4000       | 10       | 1 | Hyperbolic       | 128    | 2000    | 0           | 0.003 | 0.001  | 1    | False  | True     |
| Tirosh   | 800        | 10       | 1 | Hyperbolic       | 128    | 128     | 0           | 0.003 | 0.001  | 1    | False  | False    |
| AD       | 1500       | 20       | 1 | Hyperbolic       | 128    | 64      | 0           | 0.003 | 0.001  | 1    | False  | False    |


