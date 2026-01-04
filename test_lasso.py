'''
Test LASSO on synthetic data
'''
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
np.random.seed(0)

#%% Inputs

# Sparse observation points
N = 88


# True modes: (lambda, wd, amplitude, phase)
true_modes = [
    (5,290,1,2),
]
noise=0.1


# Candidate wavenumbers
lambdas=np.arange(1,25.1,1)#wavelength search space
wds=np.arange(180,361,5)#direction search space

#%% Functions
def lasso(x,y,f,k_search,theta_search,N_grid=100,margin=0):
    '''
    Identify dominant mode through LASSO
    '''
    from sklearn.linear_model import Lasso
    
    #make search grid
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
    lasso = Lasso(alpha=0.1, fit_intercept=False, max_iter=6000)
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
    
    X,Y=np.meshgrid(np.linspace(np.min(x)-margin,np.max(x)+margin,N_grid),np.linspace(np.min(y)-margin,np.max(y)+margin,N_grid),indexing='ij')
    f_dom=A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*X+np.sin(theta_dom)*Y)+phi[i,j])
    
    #rmse of dominant mode
    rmse=np.sum((f-A[i,j]*np.cos(k_dom*(np.cos(theta_dom)*x+np.sin(theta_dom)*y)+phi[i,j]))**2)**0.5
    
    return A, phi,k_dom,theta_dom, X, Y, f_dom,rmse

#%% Initialization
x = np.random.uniform(0, 10, N)
y = np.random.uniform(0, 10, N)

# Real signal
f = np.zeros(N)
for l, wd, A, phi in true_modes:
    k=2*np.pi/l
    th=np.radians(270-wd)
    f += A * np.cos(k*(np.cos(th)*x + np.sin(th)*y) + phi)

# Add noise
f += noise * np.random.randn(N)

#%% Main
A, phi, k_dom, theta_dom, X, Y, f_dom, rmse=lasso(x,y,f,2*np.pi/lambdas,np.radians(270-wds))
lambda_dom=2*np.pi/k_dom
wd_dom=(270-np.degrees(theta_dom))%360

#%% Plots
fig=plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
pc=plt.pcolor(X,Y,f_dom,vmin=-np.percentile(np.abs(f),90),vmax=np.percentile(np.abs(f),90),cmap='seismic')
plt.scatter(x,y,s=20,c=f,vmin=-np.percentile(np.abs(f),90),vmax=np.percentile(np.abs(f),90),cmap='seismic',edgecolor='w')
plt.colorbar(label=r"$f$")

ax = fig.add_subplot(1,2,2,projection='polar')
pcm = ax.pcolormesh(np.radians(wds), lambdas, A, shading='auto', cmap='hot')
plt.plot(np.radians(wd_dom),lambda_dom,'xg')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.show()


