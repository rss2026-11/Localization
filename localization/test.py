import numpy as np
a = np.arange(25).reshape(5,5)
obs = np.array([1, 2, 3])
scans = np.array([[0, 0, 0], [4, 4, 4]])
result = a[obs, scans]
print("a:", a)
print("result shape:", result.shape)
print("result:", result)
for i in range(2):
    for j in range(3):
        print(f"expected result[{i},{j}] = a[{obs[j]}, {scans[i, j]}] = {a[obs[j], scans[i, j]]}")
