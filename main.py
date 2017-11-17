import numpy as np
#from Small_Functions import *
from Rotation_3D import ang_vel_step3D, rot_step3D
import timeit
start = timeit.default_timer()
from Torque import torque
from Squiglys import squiglys
from Force_3D import calc_force
from RandomParticleGenerator import RandomParticleGenerator
from div0 import div0
from DistDisp import distdisp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ran_mat = lambda low, high, size: np.random.randint(low, high, size=size)
rlen = lambda x: range(len(x))
write = lambda x: ' '.join([str(x) for x in X])
vec = lambda x: np.array(X)
vlen = lambda v: np.sqrt(np.dot(v,v))
ran_mat = lambda low, high, size: np.random.randint(low,high,size=size)


"General parameters"
mu_0 = 4*np.pi
k_m = mu_0/(4 * np.pi)
inertia = 1
iterations = 1000000
timestep = 0.0001
mass = 1
radius = 0.5
image_num = 0
NumberOfParticles = 2

#pos = np.append(ran_mat(-4,4,2*vec_num),np.zeros(vec_num)).reshape(3,vec_num).transpose()
#print(pos)
#np.append(ran_mat(-4,4,2*vec_num),np.zeros(vec_num)).reshape(3,vec_num).transpose()
#pos ,vel ,m = RandomParticleGenerator(NumberOfParticles)
pos, vel, m = np.array([[-2,0,0], [2,0,0]]),  np.array([[0,0,0], [0,0,0]]),  np.array([[1,0, 0], [1,0,0]])
"Initial conditions saved for later reference"
m_0 = m
pos_0 = pos


# "Plotting parameters"
x_lim = 8.5
y_lim = 8.5
z_lim = 4.5


"Initialization"
mu_0 = np.pi*4
m_1, m_2, time  = np.array([]), np.array([]), np.array([0])
omega = np.zeros(m.shape)
pos_data, time, m_data  = pos[:,:,np.newaxis], np.array([0]),m[:,:,np.newaxis]
Sq = squiglys(pos)
dist, disp, k_m, m_i, m_j = distdisp(Sq, m, mu_0)
F = calc_force(dist, disp, k_m, m_i, m_j, mu_0)
#x_val, y_val, z_val = [], [], []
data = []

fig = plt.figure()
ax = fig.gca(projection='3d')
x,y,z = pos[:,0],pos[:,1],pos[:,2]
u,v,w = m[:,0],m[:,1],m[:,2]
ax.quiver(x,y,z,u,v,w)
ax.set_xlim([-x_lim,x_lim])
ax.set_ylim([-y_lim,y_lim])
ax.set_zlim([-z_lim,z_lim])
plt.show()
const = 0.4
for i in range(iterations):
    vel = vel + 1/2 * timestep * F/mass 
    pos = pos + timestep*vel
    Sq = squiglys(pos)
    dist, disp, k_m, m_i, m_j = distdisp(Sq, m, mu_0)
    F = calc_force(dist, disp, k_m, m_i, m_j, mu_0)
    pos = pos + timestep * vel
    tau = torque(dist, disp, m_i, m_j, mu_0, NumberOfParticles) #Calculation of torque experienced on each particle
    omega += tau / inertia * timestep 
    m = rot_step3D(m, omega, timestep) #Rotation of magnetic moments
    if dist[1,0,0] <= 2*radius:
        print("Collision Detected")
        vel = -vel#[::-1]
    #x_val.append(pos[:,0])#skal lave noget check så jeg ikke tager alle trin med
    #y_val.append(pos[:,1])
    #z_val.append(pos[:,2])
    data.append(pos)
    # if image_num == 0:
    #     pass
    # elif i % (iterations/image_num) == 0:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.set_xlim(-x_lim, x_lim)
    #     ax.set_ylim(-y_lim, y_lim)
    #     ax.set_zlim(-z_lim, z_lim)
    #     ax.plot(pos[:,0], pos[:,1], pos[:,2], 'bo')
    #     x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    #     u, v, w = m[:, 0], m[:, 1], m[:, 2]
    #     ax.quiver(x, y, z, u, v, w)
#print(data)
data = np.hstack(data)
data = [np.reshape(dat, (-1,3)) for dat in data]
stop = timeit.default_timer()
print(stop-start)
#lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

fig = plt.figure()
ax = fig.gca(projection='3d')
x,y,z = pos[:,0],pos[:,1],pos[:,2]
u,v,w = m[:,0],m[:,1],m[:,2]
ax.quiver(x,y,z,u,v,w)
ax.set_xlim([-x_lim,x_lim])
ax.set_ylim([-y_lim,y_lim])
ax.set_zlim([-z_lim,z_lim])
plt.show()

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
#for i in NumberOfParticles:
#    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
#data = [item[0] for item in data]

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax.set_xlim([-1,1])
# ax.set_ylim([-1,1])
# ax.set_zlim([-1,1])

# ax = fig.gca(projection='3d')

# x = data[0][:,0]
# y = data[0][:,1]
# z = data[0][:,2]
# N = len(z)

# #1 colored by value of `z`
# ax.scatter(x, y, z, c = plt.cm.jet(div0(z,max(z)))) 

# #2 colored by index (same in this example since z is a linspace too)

# ax.scatter(x, y, z, c = plt.cm.jet(np.linspace(0,1,N)))
# #for i in range(N-1):
# #    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(255*i/N))

# x = data[1][:,0]
# y = data[1][:,1]
# z = data[1][:,2]
# N = len(z)

# #1 colored by value of `z`
# ax.scatter(x, y, z, c = plt.cm.jet(div0(z,max(z)))) 

# #2 colored by index (same in this example since z is a linspace too)
# ax.scatter(x, y, z, c = plt.cm.jet(np.linspace(0,1,N)))
# #for i in range(N-1):
# #    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(255*i/N))

plt.show()
delta_x = data[1][:,0]
tid = np.linspace(0, iterations-2, num=iterations)
plt.scatter(tid, delta_x)
plt.show()
