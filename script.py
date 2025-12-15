import torch
from scipy.sparse.linalg import cg, LinearOperator


A = torch.eye(3)
A[2,1]  = 0.5
A[1,2]  = 0.5
b = torch.randn(50,3)
def func_A(x):
    return torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)

A_op = LinearOperator((3,3), matvec=func_A)

print(func_A(b).shape)

torch.linalg.solve(func_A, b)
x, info = torch.sparse.linalg.cg(func_A, b, atol=1e-6, maxiter=100)
print(x)
print(info)