# 用于测试各种函数
import numpy as np

a = np.array([[1, 1, 1], [2, 2, 2]])
b = np.array([1, 1, 1, 1])

print(np.array_split(b, 3))