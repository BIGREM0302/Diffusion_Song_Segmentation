import numpy as np
import matplotlib.pyplot as plt

# 假設你已經有一個高維度 array
# 這裡隨便造一個 [1000, 64, 64] 的例子，值在 [0,1] 之間
arr = np.random.rand(1000, 64, 64)

# 拉平成一維
flat_arr = arr.flatten()

# 建立 bins: 0, 0.1, 0.2, ..., 1.0
bins = np.arange(0, 1.1, 0.1)

# 統計 histogram
hist, bin_edges = np.histogram(flat_arr, bins=bins)

# 印出每個區間的數量
for i in range(len(hist)):
    print(f"{bin_edges[i]:.1f} ~ {bin_edges[i+1]:.1f}: {hist[i]}")

# 畫直方圖
plt.figure(figsize=(8,5))
plt.hist(flat_arr, bins=bins, edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Distribution of Array Values")
plt.grid(axis='y', alpha=0.75)
plt.show()