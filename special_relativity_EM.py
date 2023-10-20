# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 19:15:27 2021

@author: robin
"""
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as ani


def ro(t,x,y,z):
    """ this is the charge density, potentially time dependent"""
    # if (x**2 + y**2 + z**2)**(1/2) < 0.5:
    if x == y == z == 0:
        return - sc.elementary_charge * (2e8 + 8e7)
    else: return 0
v = 0.9 * sc.c #velocity in x
gama = 1/(1-v**2/sc.c**2)**0.5
def J(t,x,y,z):
    """ this is the independent current density"""
    
    #replace with a vector array at each point
    return ro(t,x,y,z)*v*np.array([1,0,0],float)

mu0 = sc.mu_0
e0 = sc.epsilon_0
a = 1 #grid finess
acu_t = 1e-18 # target accuracy
#we will use a hyper cube grid for simplicity
l = 8 #space time interval in consern
M = int(l / a)# this should be a range from 0 to 5 using np.arange
phi = np.zeros([M,M,M,M]) #usually the 0th index for t can be different
A = np.zeros([M,M,M,M,3]) #vector potenial in the same grid, with 0 vecotor at each point.
# phi = []
# A = []
# for t in range(M -1):
#     T = []
#     T_a = []
#     for i in range(M -1):
#         I = []
#         I_a = []
#         for j in range(M -1):
#             K = []
#             K_a = []
#             for k in range(M -1):
#                 K.append(0)
#                 K_a.append([0,0,0])
#             I.append(K)
#             I_a.append(K_a)
#         T.append(I)
#         T_a.append(I_a)
#     phi.append(T)
#     A.append(T_a)
# phi = np.array(phi,float)
# A = np.array(A,float)

#usually one would have a 4-D array for charge and current distribution instead of using
#the functions, but the essentials are the same. with a array, change ro() to ro[] and index
#are the same. but since we have functions, using arrays would cost 4 more loops, which sucks.

#gauss seidel loop, since both on the same grid, might as well use the same loop
acu = 1
N = 0
om = 0
last_diff = 1e10
while acu > acu_t:
    for t in range(1,np.shape(phi)[0] -1):
        for i in range(1,np.shape(phi)[1] -1):
            for j in range(1,np.shape(phi)[2] -1):
                for k in range(1,np.shape(phi)[3] -1):
                    #scalar potential
                    # print(t,i,j,k)
                    phi_old = np.copy(phi)
                    x = i*a - l/2
                    y = j*a - l/2
                    z = k*a - l/2
                    r = (x**2 + y**2 + z**2)**0.5
                    diff_p = (a**2*ro(t*a,x,y,z)/e0 - ((phi[t+1,i,j,k] + phi[t-1,i,j,k])/sc.c\
                    - (phi[t,i+1,j,k] + phi[t,i-1,j,k]) - (phi[t,i,j+1,k] + phi[t,i,j-1,k]) \
                        - (phi[t,i,j,k+1] + phi[t,i,j,k-1]))) /4 - phi[t,i,j,k]
                    phi[t,i,j,k] = phi[t,i,j,k] + (1+om) * diff_p
                    #vector potential
                    A_old = np.copy(A)
                    diff_A = (a**2 * mu0 * J(t*a,x,y,z) - ((A[t+1,i,j,k] +  A[t-1,i,j,k])/sc.c\
                    - (A[t,i+1,j,k] + A[t,i-1,j,k]) - (A[t,i,j+1,k] + A[t,i,j-1,k]) \
                        - (A[t,i,j,k+1] + A[t,i,j,k-1]))) /4 - A[t,i,j,k]
                    A[t,i,j,k] = A[t,i,j,k] + (1+om) * diff_A
    # acu_p = np.max(phi -  phi_old) #error comparison
    # acu_A = np.max(A -  A_old)
    # diff = max([acu_p,acu_A])
    diff = (diff_p + np.linalg.norm(diff_A))
    # if diff >= acu:
    acu = diff
    print(acu, acu_t)
    print(last_diff,diff)
    if last_diff < diff:
        print('divergent')
        break
    last_diff = diff
    N+=1
#find electric field

E = []
for t in range(np.shape(phi)[0] -1):
    E_t = [] #Efield in a specific time slice
    for i in range(np.shape(phi)[1] -1):
        E_x = [] #E in y z plane
        for j in range(np.shape(phi)[2] -1):
            E_y = [] # E in z
            for k in range(np.shape(phi)[3] -1):
                dxphi = (phi[t,i+1,j,k] - phi[t,i-1,j,k])/(2*a)
                dyphi = (phi[t,i,j+1,k] - phi[t,i,j-1,k])/(2*a)
                dzphi = (phi[t,i,j,k+1] - phi[t,i,j,k-1])/(2*a)
                delphi = np.array([dxphi,dyphi,dzphi])
                dtA = (A[t+1,i,j,k] +  A[t-1,i,j,k])/(2*a)
                e_point = -delphi - dtA
                # print(e_point)
                E_y.append(e_point)
            E_x.append(E_y)
        E_t.append(E_x)
    E.append(E_t)
E = np.array(E) #vector field E

#find B field
B = []
for t in range(np.shape(phi)[0] -1):
    B_t = [] #3D B field in a specific time slice
    for i in range(np.shape(phi)[1] -1):
        B_x = [] #B in y z plane
        for j in range(np.shape(phi)[2] -1):
            B_y = [] # B in z
            for k in range(np.shape(phi)[3] -1):
                #scalar potential
                dxAy = (A[t,i+1,j,k,1] - A[t,i-1,j,k,1])/(2*a)
                dxAz = (A[t,i+1,j,k,2] - A[t,i-1,j,k,2])/(2*a)
                dyAx = (A[t,i,j+1,k,0] - A[t,i,j-1,k,0])/(2*a)
                dyAz = (A[t,i,j+1,k,2] - A[t,i,j+1,k,2])/(2*a)
                dzAx = (A[t,i,j,k+1,0] - A[t,i,j,k-1,0])/(2*a)
                dzAy = (A[t,i,j,k+1,1] - A[t,i,j,k-1,1])/(2*a)
                B_point = np.array([dyAz - dzAy,
                            -(dxAz - dzAx),
                            dxAy - dyAx])
                B_y.append(B_point)
            B_x.append(B_y)
        B_t.append(B_x)
    B.append(B_t)
B = np.array(B)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.set_xlim3d([-l/2,l/2])
ax.set_ylim3d([-l/2,l/2])
ax.set_zlim3d([-l/2,l/2])
# plot phi for sanity check
x = np.linspace(-l/2,l/2,M)
y = np.linspace(-l/2,l/2,M)
z = np.linspace(-l/2,l/2,M)
x,y,z = np.meshgrid(x,y,z)
# ax.scatter3D(x,y,z,cmap = 'Spectral', c = phi[1])
#unfortunatly i don't know how vector fields fits here, so ploting arrows one by one is the way
# for t in range(np.shape(phi)[0] -1):
t = 3
ax.view_init(elev = 26, azim = 37)
for i in range(np.shape(phi)[1] -1):
    for j in range(np.shape(phi)[2] -1):
        for k in range(np.shape(phi)[3] -1):
            ax.quiver(i*a-l/2,j*a-l/2,k*a-l/2,E[t,i,j,k,0],\
 (E[t,i,j,k,1] - B[t,i,j,k,2] * v/sc.c)*gama,\
 (E[t,i,j,k,2] + B[t,i,j,k,1] * v/sc.c)*gama , pivot = 'tail', color = 'blue') #
            ax.quiver3D(i*a-l/2,j*a-l/2,k*a-l/2,B[t,i,j,k,0],\
gama*(B[t,i,j,k,1] + E[t,i,j,k,2] * v/sc.c),\
(B[t,i,j,k,2]- E[t,i,j,k,1] * v/sc.c)*gama ,pivot = 'tail', color = 'red') #