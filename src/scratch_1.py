import numpy as np

a = [[1, 2, 3], [4, 5, 6]]
b = np.array(a)
c = b[:, 1] * b[:, 2]

print(a)
print(type(b))
print(b)
print(type(b[0]))
print(b[0])
print(c)