import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.arange(0, 1)
y = np.arange(3, 6)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
print(Z)
print(np.zeros((3, 1)))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z)
plt.savefig('3d.png')