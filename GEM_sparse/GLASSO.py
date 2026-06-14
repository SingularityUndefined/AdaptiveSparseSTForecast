from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import time
import numpy as np
import torch


# from mingxiao
def run_glasso(X, W_gt, alpha=0.1, max_iter=30, graph_type="cycle"):
    glasso = GraphicalLasso(alpha=alpha, max_iter=max_iter)  # 你可以调整 alpha 来看看不同的稀疏程度
    tic = time.time()
    glasso.fit(X.numpy().T)  # 注意 sklearn 的输入是样本在行，特征在列
    toc = time.time()
    print(f"GLASSO fitting time: {toc - tic:.2f} seconds")

    Theta_est = glasso.precision_
    W_glasso = -Theta_est.copy()
    np.fill_diagonal(W_glasso, 0)

    dist_glasso = np.linalg.norm(W_glasso - W_gt.numpy())
    print(f"GLASSO Distance: {dist_glasso:.4f}")
    print(f"GLASSO time: {toc - tic:.2f} seconds")
    print(f"GLASSO Percentage: {dist_glasso / np.linalg.norm(W_gt.numpy()):.4f}")
    print(f"GLASSO Sparsity (nonzeros): {np.sum(W_glasso > 1e-3) / (W_gt.shape[0] * (W_gt.shape[0] - 1)):.4f}")



    plot_final_graph(torch.tensor(W_glasso, dtype=torch.float32), name="glasso_graph", title="GLASSO Learned Graph", graph_type=graph_type) 