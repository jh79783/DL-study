import  numpy as np

x_flat = np.zeros([921600,75])

k_flat= np.zeros([75,5])
conv_flat = np.matmul(x_flat, k_flat)
print(conv_flat.shape)