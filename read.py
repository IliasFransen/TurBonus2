import numpy as np
import matplotlib.pyplot as plt

import scipy.io as io

mat_data = io.loadmat('DNS_ucomp.mat')

u_comp = mat_data['u']

np.save('u_comp.npy', u_comp)