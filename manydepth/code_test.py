import numpy as np
import math
import os
import torch
import matplotlib.pyplot as plt
import glob

ratios=np.load(r'C:\Users\user\Desktop\ratios.npy')
scales=np.load(r'C:\Users\user\Desktop\scales.npy')
# print(scales)
a=ratios*scales
print(np.average(a),np.std(a))
# x=[i for i in range(1,697)]
# plt.plot(x,a)
# plt.show()