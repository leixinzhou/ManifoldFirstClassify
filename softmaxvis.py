from scipy.special import softmax
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nx, ny = (100, 100)
x = np.linspace(-10, 10, nx)
y = np.linspace(-10, 10, ny)
xv, yv = np.meshgrid(x, y)
cord = np.stack([xv,yv], axis=2)

softmax_cord = softmax(cord, axis=2)
print(cord.shape, softmax_cord.shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(xv, yv, softmax_cord[:,:,0])
ax.plot_surface(xv, yv, softmax_cord[:,:,0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()