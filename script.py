import torch
import math


def apply_chebyshev_L_inverse(b, w, func, m=5, lambda_max=None, lambda_min=None):
    # b: (B, N), approximate L^{-1} b with Chebyshev on [lambda_min, lambda_max]
    N = b.size(1)
    if lambda_max is None or lambda_min is None:
        # estimate the largest and smallest eigenvalues using power iteration
        x = torch.randn(N, device=b.device)
        for _ in range(100):
            x = func(x.unsqueeze(0), w).squeeze(0)
            x = x / torch.norm(x)
        lambda_max = (func(x.unsqueeze(0), w).squeeze(0) @ x).item()
        lambda_min = lanczos_smallest_eigenvalue(func, w, 10)  # a small positive value to avoid zero eigenvalue
        # print('Estimated eigenvalues: lambda_max=', lambda_max, ', lambda_min=', lambda_min)
        print(f"Estimated eigenvalues: lambda_max={lambda_max}, lambda_min={lambda_min}")

    a, bmax = lambda_min, lambda_max
    half = 0.5 * (bmax - a)
    center = 0.5 * (bmax + a)

    # Chebyshev coefficients for g(t)=1/(half*t+center), t in [-1,1]
    # DCT-type quadrature
    thetas = (torch.arange(m, device=b.device, dtype=b.dtype) + 0.5) * math.pi / m
    t_nodes = torch.cos(thetas)
    x_nodes = half * t_nodes + center
    g_nodes = 1.0 / x_nodes

    coeffs = torch.empty(m, device=b.device, dtype=b.dtype)
    coeffs[0] = g_nodes.mean()
    for k in range(1, m):
        coeffs[k] = (2.0 / m) * torch.sum(g_nodes * torch.cos(k * thetas))

    def apply_L_tilde(v):
        # L_tilde = (2L - (lambda_max+lambda_min)I)/(lambda_max-lambda_min)
        Lv = func(v, w)  # v: (B, N) -> (B, N)
        return (2.0 * Lv - (a + bmax) * v) / (bmax - a)

    # Chebyshev recurrence
    T0 = b
    if m == 1:
        return coeffs[0] * T0

    T1 = apply_L_tilde(b)
    out = coeffs[0] * T0 + coeffs[1] * T1

    for k in range(2, m):
        Tk = 2.0 * apply_L_tilde(T1) - T0
        out = out + coeffs[k] * Tk
        T0, T1 = T1, Tk

    return out

import torch

@torch.no_grad()
def lanczos_smallest_eigenvalue(func, w, n, k=5):
    q = torch.randn(n, device=w.device)
    q = q / q.norm()

    Q = []
    alpha = []
    beta = []

    q_prev = torch.zeros_like(q)

    for i in range(k):
        z = func(q.unsqueeze(0), w).squeeze(0)

        a = (q @ z)
        z = z - a * q - (beta[-1] * q_prev if i > 0 else 0)

        b = z.norm()

        Q.append(q)
        alpha.append(a)
        beta.append(b)

        q_prev = q
        q = z / (b + 1e-12)

    # build tridiagonal matrix
    T = torch.diag(torch.tensor(alpha)) + \
        torch.diag(torch.tensor(beta[:-1]), 1) + \
        torch.diag(torch.tensor(beta[:-1]), -1)

    eigvals = torch.linalg.eigvalsh(T)

    return eigvals[0]  # λ_min

def func(x, w):
    # a dummy function to represent the linear operator L
    # x in (B, N), w in (N, N)
    # compute w @ x for each batch
    return x @ w.t()  # (B, N) @ (N, N) -> (B, N)


w = torch.abs(torch.randn(10, 10))
w = w + w.t()  # make it symmetric
w = torch.diag(w.sum(0)) - w  # make it a Laplacian-like matrix
w = w + 0.1 * torch.ones_like(w)  # add a small constant to make it positive definite
# eigen decompose
eigenvalues, _ = torch.linalg.eig(w)
eigenvalues = eigenvalues.real
print('Eigenvalues of w:', eigenvalues)
b = torch.randn(5, 10)


alternative_result = torch.linalg.solve(w, b.t()).t()  # alternative result using torch.linalg.solve
# print('Alternative result using torch.linalg.solve:\n', alternative_result)
result = apply_chebyshev_L_inverse(b, w, func, m=5)
# print(result)

# recover b
recovered_b = func(result.unsqueeze(0), w).squeeze(0)
diff_b = torch.norm(recovered_b - b)
print('Difference between recovered b and original b:', diff_b.item())
diff = torch.norm(result - alternative_result)
print('Difference between Chebyshev approximation and torch.linalg.solve:', diff.item())