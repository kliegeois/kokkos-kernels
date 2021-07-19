import numpy as np

y_0 = np.loadtxt('y_0.txt')
y_1 = np.loadtxt('y_1.txt')
y_2 = np.loadtxt('y_2.txt')

print(np.amax(np.abs(y_0-y_1)))
print(np.amax(np.abs(y_0-y_2)))
print(np.amax(np.abs(y_1-y_2)))

print(len(y_0))
print(len(y_1))
print(len(y_2))
