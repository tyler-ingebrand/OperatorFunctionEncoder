import torch


size = 99
A = torch.rand(size, size)
A = A + A.T.conj()  # creates a Hermitian matrix

L, Q = torch.linalg.eigh(A)

A_approx = Q @ torch.diag(L) @ Q.T
print(torch.dist(A, A_approx))
