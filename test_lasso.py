import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
np.random.seed(0)

#%% Inputs

# Sparse observation points
Nobs = 250
x = np.random.uniform(0, 10, Nobs)
y = np.random.uniform(0, 10, Nobs)

# True modes: (kx, ky, amplitude, phase)
true_modes = [
    (1,1,1,2),
]
noise=0

# Real signal
f = np.zeros(Nobs)
for kx, ky, A, phi in true_modes:
    f += A * np.cos(kx*x + ky*y + phi)

# Add noise
f += noise * np.random.randn(Nobs)

# Candidate wavenumbers
k_search = np.linspace(0, 5, 51)[1:]
theta_search=np.linspace(0,2*np.pi,361)

#%% Functions
def lasso(x,y,f,k_search,theta_search,N_grid=100):
    from sklearn.linear_model import Lasso
    
    K, THETA = np.meshgrid(k_search, theta_search, indexing="ij")
    N=len(f)
    Nk = K.size
    
    #build modal matrix
    M_cos = np.zeros((N, Nk))
    M_sin = np.zeros((N, Nk))

    for i, (k, theta) in enumerate(zip(K.ravel(), THETA.ravel())):
        phase = k*(np.cos(theta)*x + np.sin(theta)*y)
        M_cos[:, i] = np.cos(phase)
        M_sin[:, i] = np.sin(phase)

    M = np.hstack([M_cos, M_sin])
    
    #LASSO optimization
    lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=6000)
    lasso.fit(M, f)

    coef = lasso.coef_

    a = coef[:Nk]      # cosine coefficients
    b = coef[Nk:]      # sine coefficients
    
    #amplitude/phase
    A = np.sqrt(a**2 + b**2).reshape(K.shape)
    phi = np.arctan2(-b, a).reshape(K.shape)
    
    #dominant mode
    ij=np.argmax(A)
    i,j=np.unravel_index(ij, A.shape)
    
    k_dom=k_search[i]
    theta_dom=theta_search[j]
    X,Y=np.meshgrid(np.linspace(np.min(x),np.max(x),N_grid),np.linspace(np.min(y),np.max(y),N_grid),indexing='ij')
    f_dom=A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*X+np.sin(theta_dom)*Y)+phi[i,j])
    
    return A, phi,k_dom,theta_dom, X, Y, f_dom

#%% Main
A, phi,k_dom,theta_dom, X, Y, f_dom=lasso(x,y,f,k_search,theta_search)

#%% Plots
fig=plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.pcolor(X,Y,f_dom,cmap='seismic',vmin=np.min(f),vmax=np.max(f))
plt.scatter(x,y,s=10,c=f,cmap='seismic',vmin=np.min(f),vmax=np.max(f),linewidth=2)
plt.colorbar(label=r"$f$")

ax = fig.add_subplot(1,2,2,projection='polar')
pcm = ax.pcolormesh(theta_search, k_search, A, shading='auto', cmap='hot')
plt.plot(theta_dom,k_dom,'xg')
plt.colorbar(pcm,label=r'$f$')
plt.show()


