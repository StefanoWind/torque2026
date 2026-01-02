import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Sparse observation points
Nobs = 250
x = np.random.uniform(0, 10, Nobs)
y = np.random.uniform(0, 10, Nobs)

# True modes: (kx, ky, amplitude, phase)
true_modes = [
    (2.0, 1, 1.5, 0),
]

# Real signal
f = np.zeros(Nobs)
for kx, ky, A, phi in true_modes:
    f += A * np.cos(kx*x + ky*y + phi)

# Add noise
f += 0.2 * np.random.randn(Nobs)

# Candidate wavenumbers
kx_grid = np.linspace(0, 6, 35)
ky_grid = np.linspace(0, 6, 35)

KX, KY = np.meshgrid(kx_grid, ky_grid, indexing="ij")
Nk = KX.size

# Dictionary: [cos | sin]
A_cos = np.zeros((Nobs, Nk))
A_sin = np.zeros((Nobs, Nk))

for i, (kx, ky) in enumerate(zip(KX.ravel(), KY.ravel())):
    phase = kx*x + ky*y
    A_cos[:, i] = np.cos(phase)
    A_sin[:, i] = np.sin(phase)

# Full real dictionary
A = np.hstack([A_cos, A_sin])

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=6000)
lasso.fit(A, f)

coef = lasso.coef_

a = coef[:Nk]      # cosine coefficients
b = coef[Nk:]      # sine coefficients

Amplitude = np.sqrt(a**2 + b**2)
Phase = np.arctan2(-b, a)

Amp_map = Amplitude.reshape(KX.shape)

ij=np.argmax(Amp_map)
i,j=np.unravel_index(ij, Amp_map.shape)

a_max=a[np.argmax(Amplitude)]
b_max=b[np.argmax(Amplitude)]
kx_max=kx_grid[i]
ky_max=ky_grid[j]

ij=np.argmax(Amp_map)
i,j=np.unravel_index(ij, Amp_map.shape)

kx_max=kx_grid[i]
ky_max=ky_grid[j]

plt.figure(figsize=(6,5))
plt.contourf(kx_grid, ky_grid, Amp_map.T, levels=30)
plt.xlabel(r"$k_x$")
plt.ylabel(r"$k_y$")
plt.title("Recovered 2D spatial spectrum (real LASSO)")
plt.colorbar(label="Amplitude")
plt.show()

X,Y=np.meshgrid(np.arange(0,10.1,0.1),np.arange(0,10.1,0.1),indexing='ij')
plt.figure()
plt.pcolor(X,Y,a_max*np.cos(kx_max * X + ky_max * Y)+b_max*np.sin(kx_max * X + ky_max * Y))
plt.scatter(x,y,s=10,c=f)
