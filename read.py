import numpy as np
import matplotlib.pyplot as plt

import scipy as sp

mat_data = sp.io.loadmat('DNS_ucomp.mat')

u_comp = mat_data['u']

print(u_comp.shape)