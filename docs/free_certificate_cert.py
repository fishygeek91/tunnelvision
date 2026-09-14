import numpy as np
rng=np.random.default_rng(1); n=6; dim=1<<n
w=np.array([bin(i).count('1') for i in range(dim)])
# H: XY hopping (weight-conserving, real symmetric) + diagonal disorder
H=np.zeros((dim,dim))
for i in range(dim):
    for a in range(n):
        for b in range(a+1,n):
            if (i>>a)&1 and not (i>>b)&1:
                j=i ^ (1<<a) ^ (1<<b)
                J=rng.normal()
                H[i,j]+=J; H[j,i]+=J
H+=np.diag(rng.normal(size=dim))
assert np.allclose(H,H.T)
# Lindblad with dephasing Z_k, gamma=0.7, t=3.0 via vectorized superop
g=0.7; t=3.0
I=np.eye(dim)
L=-1j*(np.kron(H,I)-np.kron(I,H.T))
for k in range(n):
    z=np.diag(1.0-2.0*((np.arange(dim)>>k)&1))
    L+=g*(np.kron(z,z.T)-np.kron(I,I))
from scipy.linalg import expm
E=expm(L*t)
Q=np.zeros((dim,dim))
for x in range(dim):
    rho=np.zeros((dim,dim)); rho[x,x]=1.0
    out=(E@rho.reshape(-1)).reshape(dim,dim)
    Q[:,x]=np.real(np.diag(out))
# (a) support: mass outside weight sector of x
leak=max(abs(Q[y,x]) for x in range(dim) for y in range(dim) if w[y]!=w[x])
sym=np.max(np.abs(Q-Q.T))
print(f"max cross-sector leakage = {leak:.3e}; max symmetry defect = {sym:.3e}")
