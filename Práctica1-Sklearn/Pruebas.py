import numpy as np

real = np.array([True, False, False, True])
pred = np.array([False, True, False, False])
combined = real.astype(int) - pred.astype(int)

print(real)
print(pred)
print(np.sum(combined == 1))