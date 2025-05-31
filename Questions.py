import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
plt.style.use('tableau-colorblind10')

PLOTTING = False

Viscosity = 0.00776
Length = 2*np.pi

u_comp = np.load('u_comp.npy')

Coordinates = np.linspace(-Length/2, Length/2, u_comp.shape[0])

#################################################################       Q1      #################################################################

def Correlation (Velocity_u_1: np.ndarray, Velocity_u_2: np.ndarray) -> float:

    """
    Calculate the correlation between two velocity components.

    Parameters
    ----------
    Velocity_u_1 : np.ndarray
        First velocity position, is the entire slice at that index.
    Velocity_u_2 : np.ndarray
        Second velocity position, is the entire slice at that index.
    Returns
    -------
    float
        The correlation coefficient between the two velocity components at that slice.
    """

    return np.mean(Velocity_u_1 * Velocity_u_2) 

#calculate the correlation coefficient between the velocity components at each slice in X

R11 = np.zeros((u_comp.shape[1]))
R22 = np.zeros((u_comp.shape[1]))
R33 = np.zeros((u_comp.shape[1]))

print('Calculating the correlation coefficients...')
for i in tqdm(range(u_comp.shape[1])):
    # Calculate the correlation coefficient between the velocity components at each slice in X
    R11[i] = Correlation(u_comp[:, int(u_comp.shape[1]//2), :], u_comp[:, i, :])
    R22[i] = Correlation(u_comp[int(u_comp.shape[1]//2), :, :], u_comp[i, :, :])
    R33[i] = Correlation(u_comp[:, :, int(u_comp.shape[1]//2)], u_comp[:, :, i])

if PLOTTING:
    plt.plot(Coordinates, R11)
    plt.xlabel('r')
    plt.ylabel(r'$R_{11}$')
    plt.tight_layout()
    plt.savefig('figures/R11.pdf')
    plt.show()

    plt.plot(Coordinates, R22)
    plt.xlabel('r')
    plt.ylabel(r'$R_{22}$')
    plt.tight_layout()
    plt.savefig('figures/R22.pdf')
    plt.show()

    plt.plot(Coordinates, R33)
    plt.xlabel('r')
    plt.ylabel(r'$R_{33}$')
    plt.tight_layout()
    plt.savefig('figures/R33.pdf')
    plt.show()


#################################################################       Q2      #################################################################

rho_11 = R11/ np.mean(u_comp[:, int(u_comp.shape[1]//2), :]**2)
rho_22 = R22/ np.mean(u_comp[int(u_comp.shape[1]//2), :, :]**2)
rho_33 = R33/ np.mean(u_comp[:, :, int(u_comp.shape[1]//2)]**2)

r_lam11_2 = -2*(rho_11 - 1)
r_lam22_2 = -2*(rho_22 - 1)
r_lam33_2 = -2*(rho_33 - 1)

def quad_func(x, a):
    return a * x**2

#only data for fit around 0
middle_indices = np.where(np.abs(Coordinates) < 0.05)[0]
print(f'Number of points used for fit: {len(middle_indices)}')

opt_11, cov = curve_fit(quad_func, Coordinates[middle_indices], r_lam11_2[middle_indices])
opt_22, cov = curve_fit(quad_func, Coordinates[middle_indices], r_lam22_2[middle_indices])
opt_33, cov = curve_fit(quad_func, Coordinates[middle_indices], r_lam33_2[middle_indices])

lambda_11 = np.sqrt(1/opt_11[0])
lambda_22 = np.sqrt(1/opt_22[0])   
lambda_33 = np.sqrt(1/opt_33[0])

print(f'Lambda_11: {lambda_11}')
print(f'Lambda_22: {lambda_22}')
print(f'Lambda_33: {lambda_33}')

where_2_fit = np.where(np.abs(Coordinates) < 0.25)[0]

fit_11 = 1- 0.5 * Coordinates[where_2_fit]**2 / lambda_11**2
fit_22 = 1- 0.5 * Coordinates[where_2_fit]**2 / lambda_22**2
fit_33 = 1- 0.5 * Coordinates[where_2_fit]**2 / lambda_33**2

if True:


    plt.plot(Coordinates, rho_11, label='Correlation')
    plt.plot(Coordinates[where_2_fit], fit_11, label='2nd order fit', linestyle='--')
    plt.xlabel('r')
    plt.ylabel(r'$\rho_{11}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/rho_11.pdf')
    plt.show()

    plt.plot(Coordinates, rho_22, label='Correlation')
    plt.plot(Coordinates[where_2_fit], fit_22, label='2nd order fit', linestyle='--')
    plt.xlabel('r')
    plt.ylabel(r'$\rho_{22}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/rho_22.pdf')
    plt.show()

    plt.plot(Coordinates, rho_33, label='Correlation')
    plt.plot(Coordinates[where_2_fit], fit_33, label='2nd order fit', linestyle='--')
    plt.xlabel('r')
    plt.ylabel(r'$\rho_{33}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/rho_33.pdf')
    plt.show()