"""
#GENERACIÓN DE MATRICES Y VECTORES
import numpy as np
np.__version__  
a = np.arange(10)
print(a)
b = np.arange(1, 9, 2)
print(b)
c = np.eye(3)
print(c)
d = np.diag(np.array([1, 2, 3, 4]))
print(d)
"""
"""
#GENERACIÓN DE NÚMEROS ALEATORIOS (normal)
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
alturas = np.random.normal(188, 3, 20)
print(alturas)
"""
"""
#GENERACIÓN DE NÚMEROS ALEATORIOS (binomial)
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
coinFlips = np.random.binomial(1, 0.5, 10)
print(coinFlips)
"""
"""
#Backus quiere saber cuantas botellas saldran mal tapadas si produce 100 cajas
#con una probabilidad de 1/12
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
cajas = np.random.binomial(12, 0.0833333, 100)
print(cajas)
"""
"""
#GENERACIÓN DE NÚMEROS ALEATORIOS (variado)
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s
print(np.random.uniform(size=10))
print(np.random.randint(0,12,(3,4)))
print(np.random.random_integers(1,6, size=10))

#CÁLCULO DE LA MEDIA, MEDIANA
import numpy as np
a = np.array([2,4,8,9,11,11,12])
b = np.array([2,4,8,9,11,11,120])
print ("La media arimetrica del set a es: ", np.mean(a))
print ("La media arimetrica del set b es: ", np.mean(b))
print ("Pero")
print ("El valor medio del set a es: ", np.median(a))
print ("El valor medio del set b es: ", np.median(b))
"""
"""
#GRÁFICA DE UNA LÍNEA QUEBRADA
from matplotlib import pyplot
pyplot.plot([1, 2, 3, 4], [1, 4, 9, 16])
pyplot.show()
"""

#DIAGRAMA DE DISPERSIÓN SIMPLE
import numpy as np
import matplotlib.pyplot as plt
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

"""
#HISTOGRAMA
import random
from matplotlib.pylab import hist, show
v=range(0,21)
data=[]
for i in range(1000):
    data.append(random.choice(v))
hist(data,21, (0,20))
show()
"""
"""
#DIAGRAMAS DE CAJA 1
import matplotlib.pyplot as plt
import numpy as np
data = [np.random.normal(0, std, 1000) for std in range(1, 6)]
plt.boxplot(data, notch=True, patch_artist=True)
plt.show()
import numpy as np
import pylab
data = [[np.random.rand(100)] for i in range(3)]
pylab.boxplot(data)
pylab.xticks([1, 2, 3], ['mon', 'tue', 'wed'])
pylab.show()
"""
"""
#DIAGRAMAS DE CAJA 2
from pylab import *
spread= rand(50) * 100
center = ones(25) * 50
flier_high = rand(10) * 100 + 100
flier_low = rand(10) * -100
data =concatenate((spread, center, flier_high, flier_low), 0)
boxplot(data)
figure()
boxplot(data,1)
figure()
boxplot(data,0,'gD')
figure()
boxplot(data,0,'')
figure()
boxplot(data,0,'rs',0)
figure()
boxplot(data,0,'rs',0,0.75)
spread= rand(50) * 100
center = ones(25) * 40
flier_high = rand(10) * 100 + 100
flier_low = rand(10) * -100
d2 = concatenate( (spread, center, flier_high, flier_low), 0 )
data.shape = (-1, 1)
d2.shape = (-1, 1)
data = concatenate( (data, d2), 1 )
data = [data, d2, d2[::2,0]]
figure()
boxplot(data)
show()

"""
"""
#GRÁFICOS DE VIOLIN
import random
import numpy as np
import matplotlib.pyplot as plt
fs = 10
pos = [1,2,4,5,7,8]
data = [np.random.normal(size=100) for i in pos]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,6))
axes[0, 0].violinplot(data, pos, points=20, widths=0.1,
                      showmeans=True, showextrema=True, showmedians=True)
axes[0, 0].set_title('Custom violinplot 1', fontsize=fs)
axes[0, 1].violinplot(data, pos, points=40, widths=0.3,
                      showmeans=True, showextrema=True, showmedians=True,
                      bw_method='silverman')
axes[0, 1].set_title('Custom violinplot 2', fontsize=fs)
axes[0, 2].violinplot(data, pos, points=60, widths=0.5, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
axes[0, 2].set_title('Custom violinplot 3', fontsize=fs)
axes[1, 0].violinplot(data, pos, points=80, vert=False, widths=0.7,
                      showmeans=True, showextrema=True, showmedians=True)
axes[1, 0].set_title('Custom violinplot 4', fontsize=fs)
axes[1, 1].violinplot(data, pos, points=100, vert=False, widths=0.9,
                      showmeans=True, showextrema=True, showmedians=True,
                      bw_method='silverman')
axes[1, 1].set_title('Custom violinplot 5', fontsize=fs)
axes[1, 2].violinplot(data, pos, points=200, vert=False, widths=1.1,
                      showmeans=True, showextrema=True, showmedians=True,
                      bw_method=0.5)
axes[1, 2].set_title('Custom violinplot 6', fontsize=fs)
for ax in axes.flatten():
    ax.set_yticklabels([])
fig.suptitle("Violin Plotting Examples")
fig.subplots_adjust(hspace=0.4)
plt.show()
"""
"""
#HISTOGRAMA ANIMADO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
fig, ax = plt.subplots()
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)
# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)
nverts = nrects*(1+3+1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5,0] = left
verts[0::5,1] = bottom
verts[1::5,0] = left
verts[1::5,1] = top
verts[2::5,0] = right
verts[2::5,1] = top
verts[3::5,0] = right
verts[3::5,1] = bottom
barpath = path.Path(verts, codes)
patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)
ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())
def animate(i):
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5,1] = top
    verts[2::5,1] = top
ani = animation.FuncAnimation(fig, animate, 100, repeat=False)
plt.show()
"""
"""
#USO DE LA LIBRERÍA STATISTICS
import statistics
print(statistics.mean([1, 2, 3, 4, 4]))
print(statistics.median([1, 3, 5, 7]))
print(statistics.median_low([1, 3, 5, 7]))
print(statistics.median_high([1, 3, 5, 7]))
print(statistics.mode([1, 1, 2, 3, 3, 3, 3, 4]))
print(statistics.mode(["red", "blue", "blue", "red", "green", "red", "red"]))
print(statistics.pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75]))
data = [4, 0, 2, 5, 1, 3, 4, 7]
print(statistics.pvariance(data))
mu = statistics.mean(data)
print(statistics.pvariance(data, mu))
print(statistics.mean([4, 0, 2, 5, 1, 3, 4, 7]))
print(statistics.median([4, 0, 2, 5, 1, 3, 4, 7]))
#print(statistics.stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75]))
#data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
#print(statistics.variance(data))
#m = statistics.mean(data)
#print(statistics.variance(data, m))
"""
"""
#AJUSTE MÍNIMOS CUADRADOS
import numpy as np
import statsmodels.api as sm
Y = [1,3,4,5,2,3,4]
X = range(1,8)
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.params
results.tvalues
print(results.t_test([1, 0]))
"""




























