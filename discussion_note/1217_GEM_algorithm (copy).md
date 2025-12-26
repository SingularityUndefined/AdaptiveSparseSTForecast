# Discussion Dec 17

## 1 GEM Algorithm (Sparse Version)

Notations: $\mathbf{S}=\frac{1}{N}\sum_{k=1}^{N}\mathbf{x}^{(k)}\mathbf{x}^{(k)T}$, learned graph weight matrix $\mathbf{W}^o$, adjacency mask $\mathbf{A}$, actual graph weight matrix $\mathbf{W}$, Laplacian matrix $\mathbf{L}$

### 0) Graph Learning Module

- Learnable embeddings + 1-layer MLP (LeakyReLU activation)
- Gaussian kernel weight generation: $w_{ij}^o = \alpha\exp(-\Vert \mathbf{f}_i-\mathbf{f}_j\Vert_2^2/2\theta)$, where $\theta$ is learnable, $\alpha$ is just a scale controller for PGD.

### 1) E-step

$$
\min_\mathbf{x} \Vert \mathbf{y}-\mathbf{x}\Vert_2^2+\mu\mathbf{x}^\top\mathbf{Lx}\quad \Rightarrow\quad \mathbf{x}=(\mathbf{I}+\mu\mathbf{L})^{-1}\mathbf{y}
$$

Solve by **CG**

### 2) M-step-1

$$
\mathcal{L}(\theta) = \text{tr}(\mathbf{SL}) - \log|\mathbf{L}| = \text{tr}(\mathbf{SL}) - \log\det(\mathbf{L}+\mathbf{J})\\
$$

$\text{tr}(\mathbf{SL})=\frac 1 N \sum_{i=1}^N\mathbf{x^{(i)}}^\top\mathbf{Lx}=\frac 1 N \sum_{i=1}^N \sum_{(u,v)\in\mathcal{E}}w_{uv}(x_u^{(i)}-x_v^{(i)})^2$

#### Challenge 1: Gradient Step

$$
\nabla_\theta\log\det(\mathbf{L}+\mathbf{J})|_{\theta=\theta_0} = [\mathbf{L}(\theta,\mathbf{A}) + \mathbf{J})^{-1}\nabla_{\theta}\mathbf{L}(\theta,\mathbf{A})]|_{\theta=\theta_0}=(\mathbf{L}(\theta_0,\mathbf{A}) + \mathbf{J})^{-1}\nabla_\theta\mathbf{L}(\theta,\mathbf{A}))|_{\theta=\theta_0}\\
=\nabla_\theta\text{tr}((\mathbf{L}(\theta_0,\mathbf{A})+\mathbf{J})^{-1}\mathbf{L}(\theta, \mathbf{A}))|_{\theta=\theta_0}\\
\tilde{\mathcal{L}}(\theta) = \text{tr}(\mathbf{SL}(\theta)) - \text{tr}(\mathbf{R}_0\mathbf{L}(\theta)),\quad \mathbf{R}_0:=(\mathbf{L}(\theta_0)+\mathbf{J})^{-1}
$$

Constraints: $\Vert \mathbf{L} \Vert_F = c$.

***Note:** $\text{tr}(\mathbf{R}_0\mathbf{L}(\theta))|_{\theta=\theta_0}=\text{tr}((\mathbf{L}(\theta_0)+\mathbf{J})^{-1}\mathbf{L}(\theta_0))=\text{tr}((\mathbf{L}^\dagger(\theta_0)+\mathbf{J})\mathbf{L}(\theta_0))=n-1$ is a constant. But we have to compute it in a matrix multiplication form to **preserve the gradient** at $\mathbf{L}$, while making $\mathbf{R}_0$ a **non-differentiable coefficient.***

**[Algorithm M1]** Therefore, the proximal gradient descent method in M-step-1:

1. Compute $\mathbf{R}_0 = (\mathbf{L}+\mathbf{J})^{-1}$ with **no grad**
2. Compute the scalar $\tilde{\mathcal{L}}(\theta) = \text{tr}((\mathbf{S}-\mathbf{R}_0)\mathbf{L})$, since $\nabla_\theta\tilde{\mathcal{L}}(\theta)=\nabla_\theta \mathcal{L}(\theta)$
3. Use `autograd` to compute the gradients $\nabla{\theta}$
4. Gradient step: $\theta^{\text{new}} = \theta - \eta\nabla_\theta \tilde{\mathbf{L}}(\theta)$, compute corresponding $\mathbf{W}^o,\mathbf{W}, \mathbf{L}$
5. Rescale $\mathbf{L}$: update $\alpha = c/\Vert\mathbf{L}\Vert_F$, update rescaled $\mathbf{W}^o$ at the same time

#### Challenge 2: Inverse of the Laplacian Matrix

Note that $\mathbf{L}$ is actually sparse, we only need the entries in $\mathbf{R}_0$ in the same position of $\mathbf{L}$: the **diagonals** and the **neighbors** defined by $\mathbf{A}$
$$
\text{tr}((\mathbf{S}-\mathbf{R}_0)\mathbf{L}) = \sum_{i=j\text{ or }A_{i,j}=1}(S_{i,j}-{R_{0}}_{i,j})L_{i,j},\quad\text{tr}(\mathbf{SL})=\frac{1}{N}\sum_{k=1}^{N}\mathbf{x}^{(k)\top}\mathbf{L}\mathbf{x}^{(k)}=\frac{1}{N}\sum_{k=1}^N \sum_{(i,j)\in\mathcal{E}}W_{ij}(x_i^{(k)}-x_j^{(k)})^2\\
\text{tr}(\mathbf{R}_0\mathbf{L}) = \text{tr}\left(\mathbf{R}_0\sum_{(u,v)\in\mathcal{E}}W_{uv}(\mathbf{e}_u-\mathbf{e}_v)(\mathbf{e}_u-\mathbf{e}_v)^\top\right)=\sum_{(u,v)\in\mathcal{E}}W_{uv}(\mathbf{e}_u-\mathbf{e}_v)^\top \mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)
$$

We only need to solve $\mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)=(\mathbf{L}+\mathbf{J})^{-1}(\mathbf{e}_u-\mathbf{e}_v)$ for each edge.

Solve $(\mathbf{L}+\mathbf{J})\mathbf{z}=\mathbf{e}_u-\mathbf{e}_v$ for each $(u,v)\in\mathcal{E}$ $\Leftrightarrow$ solve $\mathbf{L}\mathbf{z}=\mathbf{e}_u-\mathbf{e}_v$ in $\mathbf{1}^\perp$, since $\mathbf{1}^\top(\mathbf{e}_u-\mathbf{e}_v)=0$. Then $w_{uv}(\mathbf{e}_u-\mathbf{e}_v)^\top \mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)=w_{uv}(z_u-z_v)$

Note that $\mathbf{R}_0$ **does not require gradients,** and so does $\mathbf{L}$ in this section. 

(1) $|\mathcal{E}|=k|\mathcal{V}|=nk$ could be large, conjugated gradients could be expensive. (2) The matrix is fixed for all edges to compute, and does not require gradient.

**Method 1**: Use ***sparse Cholesky Decomposition*** by `sksparse.cholmod.cholesky`  ($\mathbf{LDL}^\top$ for PSD matrices)

If PSD matrix decomposition is not supported: $\mathbf{z} = \mathbf{L}^\dagger(\mathbf{e}_u-\mathbf{e}_v)\approx [(\mathbf{L}+\epsilon\mathbf{I})^{-1} - \frac{1}{\epsilon n}\mathbf{11}^\top](\mathbf{e}_u-\mathbf{e}_v)=(\mathbf{L}+\epsilon\mathbf{I})^{-1}(\mathbf{e}_u-\mathbf{e}_v)$

**Method 2**: Solve $|\mathcal{E}|$ linear equations by **CG**.

#### *Update Formula*

$$
\tilde{\mathcal{L}}(\theta)=\sum_{(i,j)\in\mathcal{E}} W_{ij}^o(\theta)A_{ij}\left[\frac{1}{N}\sum_{k=1}^{N}(x_i^{(k)}-x_j^{(k)})^2 - (\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{R}_0 (\mathbf{e}_i-\mathbf{e}_j)\right],\quad \theta^{\text{new}}=\theta-\delta \nabla_\theta\tilde{\mathbf{L}}(\theta)
$$

### 3) M-step-2

Relaxed optimization:
$$
\min_{\mathbf{A}\in [0,1]^{n\times n},\mathbf{A}=\mathbf{A}^\top}\text{tr}(\mathbf{SL}) - \log\det(\mathbf{L}+\mathbf{J}) + \gamma \Vert\mathbf{A} \Vert_{1,\text{off}} \quad\text{s.t. } \mathbf{W}=\mathbf{W}^o\circ \mathbf{A},\mathbf{L}=\text{diag}(\mathbf{W1})-\mathbf{W}
$$
**[Algorithm M2] Proximal Gradient Descent:**

1. Compute new $\mathbf{R} = (\mathbf{L} + \mathbf{J})^{-1}$ from the updated $\mathbf{L}$
2. Compute $\tilde{\mathbf{R}}=\mathbf{R}-\frac{1}{2}(\mathbf{r1}^\top +\mathbf{1r}^\top),\tilde{\mathbf{S}}=\mathbf{S}-\frac{1}{2}(\mathbf{s1}^\top+\mathbf{1s}^\top)$, where $\mathbf{r}=\text{diag}(\mathbf{R}),\mathbf{s}=\text{diag}(\mathbf{S})$
3. PGD step: $\mathbf{A}^{\text{new}}=\Pi_{[0,1]^{N\times N}}[S_{\eta \gamma}(\mathbf{A}-\eta \mathbf{W}^o\circ(\tilde{\mathbf{R}}-\tilde{\mathbf S}))]$
4. Recompute graph $\mathbf{W}^\text{new}=\mathbf{W}^o \circ \mathbf{A}^{\text{new}}$, rescale to $\Vert \mathbf{L}\Vert_F=c$

#### Challenge: compute $\tilde{\mathbf{R}}\circ \mathbf{W}$ *with* gradient

$\tilde{R}_{i,j}=-\frac1 2(\mathbf{e}_i-\mathbf{e}_j)^\top (\mathbf{L}+\mathbf{J})^{-1}(\mathbf{e}_i-\mathbf{e}_j)=-\frac 1 2(R_{ii}+R_{jj}-2R_{ij})=-\frac1 2(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)$      ($\mathbf{R}=\mathbf{L}^\dagger+\mathbf{J}$)

$(\tilde{\mathbf{R}}\circ\mathbf{W}^o)_{ij}=(\tilde{\mathbf{R}}\circ\mathbf{W}^o)_{ji}=-\frac{1}{2}W^o_{ij}(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)$

For each edge, solve $\mathbf{Lz}=\mathbf{e}_i-\mathbf{e}_j$ in $\mathbf{1}^\perp$, then compute $-\frac{1}{2}w_{ij}^o(z_i-z_j)$

<font color=red>**(preserve gradient?)**</font> Solve by **CG**. 



Compute $\tilde{\mathbf{S}}\circ \mathbf{W}^o$

$(\tilde{\mathbf{S}}\circ \mathbf{W}^o)_{ij}=w^o_{ij}(S_{ij}-\frac 1 2 S_{ii}-\frac 1 2 S_{jj})=\frac 1 N w_{ij}^o\sum_{k=1}^N[ x_i^{(k)}x_j^{(k)}-\frac 1 2 (x_i^{(k)2}+x_j^{(k)2})]=-\frac 1 {2N} W_{ij}^o \sum_{k=1}^N (x_i^{(k)}-x_j^{(k)})^2$



#### *Update Formula*

$$
A_{ij}^{\text{new}} = \Pi_{[0,1]}\left[S_{\eta\gamma}\left(A_{ij}-\eta W^o_{ij}\left(\frac{1}{N}\sum_{k=1}^{N}(x_i^{(k)}-x_j^{(k)})^2 -(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j) \right)\right)\right]
$$

### *Discussion*

1. The two updating formulas are closely correlated; the differences are (1) the summation (2) the usage of $W\circ A$ or $W^o$ (3) whether to keep the gradients of $\mathbf{L}^\dagger$
2. Better solution for term $(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)$ for all $(i,j)\in\mathcal{E}$?

## 2 Complexity Analysis

$n$ nodes, degree $d$

operation $\mathbf{x}\to\mathbf{Lx}$: $\mathcal{O}(nd)$

- E-step (unrolled CG): $\mathcal{O}(ndT)$ ($T$: unrolled CG iterations)
- M-step 1
  - unrolled (block) CG: $\mathcal{O}(nd\cdot T\cdot nd)$ compute for $\frac{nk}2$ edges
  - Sparse Cholesky Decomposition (2D grid): $\mathcal{O}(nd\log n)+\mathcal{O}(n^{1.5})+\mathcal{O}(nd\cdot nd)$ (symbolic factorization + Cholesky decompsition + solve $nk/2$ linear systems)
- M-step 2
  - unrolled CG: $\mathcal{O}(Tn^2d^2)$

Question: Other simpler method to solve $w^o_{ij}(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)$ for all $(i,j)\in\mathcal {E}$? In unrolled CG, we need to solve $|\mathcal{E}|$ linear equations at the same time.

## 3 Implementation

Use an edge weight list to save all weights by all edges,sparse matrix multiplication: 
$$
(\mathbf{Lx})_i = \sum_{j\in\mathcal{N}_i}w_{ij}(x_i-x_j),\quad (\mathbf{(L+J)x})_i = \sum_{j\in\mathcal{N}_i}w_{ij}(x_i-x_j)+\bar{x}
$$
