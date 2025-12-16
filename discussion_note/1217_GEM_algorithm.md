# Discussion Dec 17

## 1 GEM Algorithm (Sparse Version)

### 0) Graph Learning Module

- Learnable embeddings + 1-layer MLP (LeakyReLU activation)
- Gaussian kernel weight generation: $w_{ij}^o = \alpha\exp(-\Vert \mathbf{f}_i-\mathbf{f}_j\Vert_2^2/2\theta)$, where $\theta$ is learnable, $\alpha$ is just a scale controller for PGD.

### 1) E-step

$$
\min_\mathbf{x} \Vert \mathbf{y}-\mathbf{x}\Vert_2^2+\mu\mathbf{x}^\top\mathbf{Lx}\quad \Rightarrow\quad \mathbf{x}=(\mathbf{I}+\mu\mathbf{L})^{-1}\mathbf{y}
$$

Solve with Conjugated gradients

### 2) M-step-1

$$
\mathcal{L}(\theta) = \text{tr}(\mathbf{SL}) - \log|\mathbf{L}| = \text{tr}(\mathbf{SL}) - \log\det(\mathbf{L}+\mathbf{J})\\
$$

$\text{tr}(\mathbf{SL})=\frac 1 N \sum_{i=1}^N\mathbf{x^{(i)}}^\top\mathbf{Lx}=\frac 1 N \sum_{i=1}^N \sum_{(u,v)\in\mathcal{E}}w_{uv}(x_u^{(i)}-x_v^{(i)})^2$

#### Challenge 1: Gradient Step

$$
\nabla_\theta\log\det(\mathbf{L}+\mathbf{J})|_{\theta=\theta_0} = [\mathbf{L}(\theta,\mathbf{A}) + \mathbf{J})^{-1}\nabla_{\theta}\mathbf{L}(\theta,\mathbf{A})]|_{\theta=\theta_0}=(\mathbf{L}(\theta_0,\mathbf{A}) + \mathbf{J})^{-1}\nabla_\theta\mathbf{L}(\theta,\mathbf{A}))|_{\theta=\theta_0}=\nabla_\theta\text{tr}((\mathbf{L}(\theta_0,\mathbf{A})+\mathbf{J})^{-1}\mathbf{L}(\theta, \mathbf{A}))|_{\theta=\theta_0}\\
\tilde{\mathcal{L}}(\theta) = \text{tr}(\mathbf{SL}(\theta)) - \text{tr}(\mathbf{R}_0\mathbf{L}(\theta)),\quad \mathbf{R}_0:=(\mathbf{L}(\theta_0)+\mathbf{J})^{-1}
$$

Constraints: $\Vert \mathbf{L} \Vert_F = c$.

> Note: $\text{tr}(\mathbf{R}_0\mathbf{L}(\theta))|_{\theta=\theta_0}=\text{tr}((\mathbf{L}(\theta_0)+\mathbf{J})^{-1}\mathbf{L}(\theta_0))=\text{tr}((\mathbf{L}^\dagger(\theta_0)+\mathbf{J})\mathbf{L}(\theta_0))=n-1$ is a constant. But we have to compute it in a matrix multiplication form to **preserve the gradient** at $\mathbf{L}$, while making $\mathbf{R}_0$ a **non-differentiable coefficient.**

**[Algorithm M1]** Therefore, the proximal gradient descent method in M-step-1:

1. Compute $\mathbf{R}_0 = (\mathbf{L}+\mathbf{J})^{-1}$ with **no grad**
2. Compute the scalar $\tilde{\mathcal{L}}(\theta) = \text{tr}((\mathbf{S}-\mathbf{R}_0)\mathbf{L})$, since $\nabla_\theta\tilde{\mathcal{L}}(\theta)=\nabla_\theta \mathcal{L}(\theta)$
3. Use `autograd` to compute the gradients $\nabla{\theta}$
4. Gradient step: $\theta^{\text{new}} = \theta - \eta\nabla\theta$, compute corresponding $\mathbf{W}^o,\mathbf{W}, \mathbf{L}$
5. Rescale $\mathbf{L}$: update $\alpha = c/\Vert\mathbf{L}\Vert_F$, update rescaled $\mathbf{W}^o$ at the same time

#### Challenge 2: Inverse of the Laplacian Matrix

Note that $\mathbf{L}$ is actually sparse, we only need the entries in $\mathbf{R}_0$ in the same position of $\mathbf{L}$: the **diagonals** and the **potential neighbors
$$
\text{tr}((\mathbf{S}-\mathbf{R}_0)\mathbf{L}) = \sum_{i=j\text{ or }A_{i,j}=1}(S_{i,j}-{R_{0}}_{i,j})L_{i,j}=\sum_i D_{i}(S_{i,i}-{R_{0}}_{i,i}) - \sum_{(i,j)\in\mathcal{E}(\mathbf{A})}(S_{i,j}-{R_{0}}_{i,j})W_{i,j}=\sum_{(i,j)\in\mathcal{E}(\mathbf{A})}W_{i,j}[(S_{i,i}-S_{i,j})-({R_{0}}_{i,i}-{R_{0}}_{i,j})]
$$

> ##### ~~Method 1: Neumann Series~~
>
> ***1) Unshifted CGL*** Use *Neumann-Series* to compute $\mathbf{R}_0$: since $\Vert \mathbf{L} \Vert_F = c = \sqrt{\sum_i \lambda_i^2} > \lambda_{\text{max}}$. $\Vert \mathbf{L}+\mathbf{J}\Vert_F=\sqrt{1+\sum_{i\ge 2}\lambda_i^2}>\lambda_{\text{max}}$.
> $$
> \mathbf{R}_0 \approx \alpha \sum_{k=0}^{K} (\mathbf{I} -\alpha (\mathbf{L}+\mathbf{J}))^k,\quad 0<\alpha<\frac{2}{\lambda_{\text{max}}}
> $$
> let $\alpha = 2/c$, and set $K=5$ or $K=10$.
>
> **Detailed implementation:** denote $\mathbf{M}=\mathbf{I}-\alpha(\mathbf{L}+\mathbf{J})$, $\mathbf{R}_0 = \mathbf{I} + \mathbf{M}(\mathbf{I} +\mathbf{M}(\mathbf{I}+\cdots))$
>
> For any symmetric matrix $\mathbf{B}$,
> $$
> (\mathbf{MB})_{i,j}=\sum_k M_{i,k}B_{k,j}=B_{i,j} -\alpha \sum_k(L_{i,k} + \frac 1 n)B_{j,k}=B_{i,j}-\frac{\alpha}{n}\mathbf{1}^\top\mathbf{b}_j-\alpha\sum_{k\in\mathcal{N}_i}L_{i,k}B_{j,k}
> $$
> ***2) Shifted CGL Estimation (Dangerous)***
> $$
> \min_\theta \text{tr}(\mathbf{S}\mathbf{L}) - \log\det(\mathbf{L}+\epsilon\mathbf{I})+\gamma\Vert\mathbf{L}\Vert_{0,\text{off}},\quad \text{s.t. }\dots
> $$
> <font color=red>**Potential Danger:**</font> $\mathbf{L} = \mathbf{P} \text{diag}(0,\lambda_2,\dots, \lambda_n)\mathbf{P}^\top$, $(\mathbf{L}+\epsilon\mathbf{I})^{-1} = \mathbf{P}\text{diag}(\frac{1}{\epsilon},\frac{1}{\lambda_2+\epsilon},\dots ,\frac{1}{\lambda_n+\epsilon})\mathbf{P}^\top$, <font color=red>$1/\epsilon$ could be large.</font>
>
> While $(\mathbf{L}+\mathbf{J})^{-1}=\mathbf{P} \text{diag}(1,\frac1 {\lambda_2},\dots, \frac 1{\lambda_n})\mathbf{P}^\top=\mathbf{L}^\dagger + \mathbf{J}$. Therefore, <font color=red>$\mathbf{L}^\dagger =\lim_{\epsilon\rightarrow 0} (\mathbf{L}+\epsilon\mathbf{I})^{-1} -\frac{1}{\epsilon} \mathbf{J}$</font>
>
> Define $\mathbf{R}_0 = (\mathbf{L}+\epsilon\mathbf{I})^{-1}\approx \alpha\sum_{k=0}^K(\mathbf{I}-\alpha (\epsilon\mathbf{I}+\mathbf{L}))^k,\alpha\le \frac{1}{\epsilon+c}$.
>
> Let $\alpha=\frac{1}{\epsilon+c}$, 
> $$
> \mathbf{R}_0 = \frac{1}{\epsilon+c}\sum_{k=0}^K \left(\frac{c}{\epsilon+c}\mathbf{I} - \frac{1}{\epsilon+c}\mathbf{L}\right)^k
> $$
> (Alternative: $(\mathbf{L}+\mathbf{J})^{-1}=\mathbf{L}^\dagger+\mathbf{J}=\lim_{\epsilon\rightarrow0^+}\alpha\sum_{k=0}^\infty(\mathbf{I}-\alpha (\epsilon\mathbf{I}+\mathbf{L}))^k$)
>
> **Problem:** In both methods, the inverse has to be solved in the original form of $n\times n$. Sparse computation is not applicable.

##### Method 2 Linear Equation Solver

$$
\text{tr}(\mathbf{R}_0\mathbf{L}) = \text{tr}\left(\mathbf{R}_0\sum_{(u,v)\in\mathcal{E}}w_{uv}(\mathbf{e}_u-\mathbf{e}_v)(\mathbf{e}_u-\mathbf{e}_v)^\top\right)=\sum_{(u,v)\in\mathcal{E}}w_{uv}(\mathbf{e}_u-\mathbf{e}_v)^\top \mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)
$$

We only need to solve $\mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)=(\mathbf{L}+\mathbf{J})^{-1}(\mathbf{e}_u-\mathbf{e}_v)$ for each edge.

Solve $(\mathbf{L}+\mathbf{J})\mathbf{z}=\mathbf{e}_u-\mathbf{e}_v$ for each $(u,v)\in\mathcal{E}$ $\Leftrightarrow$ solve $\mathbf{L}\mathbf{z}=\mathbf{e}_u-\mathbf{e}_v$ in $\mathbf{1}^\perp$, since $\mathbf{1}^\top(\mathbf{e}_u-\mathbf{e}_v)=0$. Then $w_{uv}(\mathbf{e}_u-\mathbf{e}_v)^\top \mathbf{R}_0(\mathbf{e}_u-\mathbf{e}_v)=w_{uv}(z_u-z_v)$

Note that $\mathbf{R}_0$ **does not require gradients,** and so does $\mathbf{L}$ in this section. 

(1) $|\mathcal{E}|=k|\mathcal{V}|=nk$ could be large, conjugated gradients could be expensive. (2) The matrix is fixed for all edges to compute, and does not require gradient.

Use *sparse Cholesky Decomposition* by `sksparse.cholmod.cholesky`  ($\mathbf{LDL}^\top$ for PSD matrices)

If PSD matrix decomposition is not supported: $\mathbf{z} = \mathbf{L}^\dagger(\mathbf{e}_u-\mathbf{e}_v)\approx [(\mathbf{L}+\epsilon\mathbf{I})^{-1} - \frac{1}{\epsilon n}\mathbf{11}^\top](\mathbf{e}_u-\mathbf{e}_v)=(\mathbf{L}+\epsilon\mathbf{I})^{-1}(\mathbf{e}_u-\mathbf{e}_v)$

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

$\tilde{R}_{i,j}=-\frac 1 2(R_{ii}+R_{jj}-2R_{ij})=-\frac1 2(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)=-\frac1 2(\mathbf{e}_i-\mathbf{e}_j)^\top (\mathbf{L}+\mathbf{J})^{-1}(\mathbf{e}_i-\mathbf{e}_j)$      ($\mathbf{R}=\mathbf{L}^\dagger+\mathbf{J}$)

$(\tilde{\mathbf{R}}\circ\mathbf{W}^o)_{ij}=(\tilde{\mathbf{R}}\circ\mathbf{W}^o)_{ji}=-\frac{1}{2}w^o_{ij}(\mathbf{e}_i-\mathbf{e}_j)^\top \mathbf{L}^\dagger(\mathbf{e}_i-\mathbf{e}_j)$

For each edge, solve $\mathbf{Lz}=\mathbf{e}_i-\mathbf{e}_j$ in $\mathbf{1}^\perp$, then compute $-\frac{1}{2}w_{ij}^o(z_i-z_j)$

<font color=red>**(preserve gradient?)**</font> Solve by CG. 

## 2 Complexity Analysis

$n$ nodes, degree $k$

operation $\mathbf{x}\to\mathbf{Lx}$: $\mathcal{O}(nk)$

- E-step (unrolled CG): $\mathcal{O}(nkT)$ ($T$: unrolled CG iterations)
- M-step 1
  - unrolled (block) CG: $\mathcal{O}(nk\cdot T\cdot nk)$ compute for $\frac{nk}2$ edges
  - Sparse Cholesky Decomposition (2D grid): $\mathcal{O}(nk\log n)+\mathcal{O}(n^{1.5})+\mathcal{O}(nk\cdot nk)$ (symbolic factorization + Cholesky decompsition + solve $nk/2$ linear systems)
- M-step 2
  - unrolled CG: $\mathcal{O}(Tn^2k^2)$

## 2 Experiments