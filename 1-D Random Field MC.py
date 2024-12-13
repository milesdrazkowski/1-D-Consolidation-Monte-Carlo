# This code implements a Monte Carlo simulation framework to study the average degree of
# consolidation, focusing on spatial variability and its effects.
# It uses a combination of numerical methods and analytical approximations (NAVFAC) to evaluate
# consolidation behavior and includes data visualization to compare the results. This
# code provides a probablistic comparison of the NAVFAC and Monte Carlo random field
# implementation.

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
def lognormal_params(median, cv):
    mu = np.log(median)
    sigma = np.sqrt(np.log(cv**2 + 1))
    return mu, sigma

def nearest_positive_semidefinite(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 0)  # Set negative eigenvalues to 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def gen_rackwitz_corr_matrix(elmnts, size, theta):
    indices = np.arange(elmnts)
    corr_matrix = np.exp(-2*size*np.abs(indices[:, None] - indices[None, :]) / theta)
    # Ensure the matrix is positive semi-definite
    corr_matrix = nearest_positive_semidefinite(corr_matrix)
    return corr_matrix

def gen_corr_matrix(mu, sigma, theta, elmnts, size):
    corr_matrix = gen_rackwitz_corr_matrix(elmnts, size, theta)
    cov_matrix = corr_matrix * (sigma**2)
    mu_vector = np.full(elmnts, mu)
    normal_samples = np.random.multivariate_normal(mu_vector, cov_matrix)
    lognormal_samples = np.exp(normal_samples)
    return lognormal_samples

# Monte Carlo parameters
num_simulations = 100000  # Number of Monte Carlo iterations

INDICT = 0
rr = []

# Time step parameters
dt = 1 / 12
tf = 10
steps = int(tf / dt)
time = np.arange(steps) * dt  # Time array

u0 = 1000 # Initial Load

# Storage for results
cv_results = []
bottom_U_results = np.zeros((num_simulations, steps))  # Degree of consolidation at the bottom node
Unavfac_results = np.zeros((num_simulations, steps))

# Log-normal distribution parameters for cv and L
cv_median = 15
cv_cv = 1.2
L_median = 5
L_cv = 0.9
cv_mu, cv_sigma = lognormal_params(cv_median, cv_cv)
cv_VAR = cv_sigma ** 2
L_mu, L_sigma = lognormal_params(L_median, L_cv)

# Monte Carlo loop
for sim in range(num_simulations):
    # Randomize parameters
    layers = np.random.randint(2, 5)  # Number of layers (e.g., 2 to 4 layers)
    n = np.random.randint(10, 50, size=layers)  # Elements per layer (10 to 50)
    L = np.random.lognormal(L_mu, L_sigma, size=layers)  # Log-normal distribution for L

    I = L / n  # Element sizes
    H = sum(L)  # Total height
    nt = sum(n)  # Total number of elements

    cv=[]
    mus=[]
    for layer in range(layers):
        nelmnts = n[layer]
        theta = L[layer]*np.random.beta(5, 2)
        mu = np.random.uniform(cv_mu-(0.05*cv_sigma),cv_mu+(0.5*cv_sigma))
        sigma = np.random.uniform(0.95*cv_sigma,1.05*cv_sigma)
        corr_cv = gen_corr_matrix(mu, sigma, theta, nelmnts, I[layer])
        cv.extend(corr_cv)
        mus.append(mu)

    mus = np.array(mus)
    cv = np.array(cv)
    cv_results.append(cv)

    Heff = L[0]
    for i in range(1,layers-1,1):
        Heff = Heff + (L[i] * math.sqrt(mus[i] / mus[0]))

    # Initialize matrices
    K = np.zeros((nt + 1, nt + 1))
    KT = np.zeros((nt + 1, nt + 1))
    RS = np.zeros((nt + 1, steps))
    u = np.zeros((nt + 1, steps))
    U = np.zeros((nt + 1, steps))
    u[:, 0] = u0
    U[:, 0] = u0

    # Precompute constants
    I_dt = I / dt

    # Assemble global matrices
    for j in range(len(n)):
        count = n[j]
        for i in range(count):
            idx = i + sum(n[:j])
            stiffness = (cv[idx] / I[j]) * np.array([[1, -1], [-1, 1]])  # Element stiffness matrix
            K[idx:idx + 2, idx:idx + 2] += stiffness
            KT[idx:idx + 2, idx:idx + 2] += (I_dt[j] / 6) * np.array([[2, 1], [1, 2]])

    Ktot = K + KT
    tempKtot = Ktot[1:-1, 1:-1]
    invKtot = np.linalg.inv(tempKtot)

    # Time-stepping loop
    for t in range(steps - 1):
        RS[1:-1, t] = np.dot(KT[1:-1, :], u[:, t])
        u[1:-1, t + 1] = np.dot(invKtot, RS[1:-1, t])

    for j in range(len(n)):
        count = n[j]
        for i in range(count):
            idx = i + sum(n[:j])
            if idx == 0:
                U[idx, :] = (0.5 * I[j] * (u[idx, :] + u[idx + 1, :]))
            else:
                U[idx, :] = (0.5 * I[j] * (u[idx, :] + u[idx + 1, :])) + U[idx - 1, :]

    U[nt, :] = (0.5 * I[-1] * u[nt, :]) + U[nt - 1, :]
    U = ((u0 * H) - U) / (u0 * H)

    # Store results for the bottom node
    bottom_U_results[sim, :] = U[nt, :]

    T = mus[0]*time/((Heff/2)**2)
    Unavfac = (T**3/(T**3+0.5))**(1/6)
    Unavfac_results[sim,:] = Unavfac

    INDICT = INDICT + np.sum((U[nt,:] < Unavfac))/steps
    rr.append(np.corrcoef(U[nt,:], Unavfac)[0, 1])

mean_U = np.mean(bottom_U_results, axis=0)
std_U = np.std(bottom_U_results, axis=0)
mean_Unavfac = np.mean(Unavfac_results, axis=0)
std_Unavfac = np.std(Unavfac_results, axis=0)

mse = mean_squared_error(mean_U, mean_Unavfac)
rmse = np.sqrt(mse)

rr = np.array(rr)
correlation = np.mean(rr)

absolute_difference = np.abs(mean_U - mean_Unavfac)
relative_difference = np.abs((mean_U - mean_Unavfac) / np.maximum(mean_U, 1e-6))

Unavfac_percentile = INDICT/num_simulations

print(f"Maximum absolute difference: {np.max(absolute_difference):.4f}")
print(f"Maximum relative difference: {np.max(relative_difference):.4%}")
print(f"Percentile: {Unavfac_percentile:.4f}")

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Correlation: {correlation:.4f}")

threshold = 0.25
time_U = time[np.argmax(mean_U >= threshold)]
time_Unavfac = time[np.argmax(mean_Unavfac >= threshold)]

print(f"Time to reach 50% consolidation:")
print(f"Mean U: {time_U:.2f} years")
print(f"Mean Unavfac: {time_Unavfac:.2f} years")

plt.figure(figsize=(8, 5))
plt.hist(bottom_U_results[:, -1], bins=20, alpha=0.75, color='blue', edgecolor='black', label="Final U")
plt.hist(Unavfac_results[:, -1], bins=20, alpha=0.75, color='red', edgecolor='black', label="Final Unavfac")
plt.title("Histogram of Final Average Degree of Consolidation")
plt.xlabel("Average Degree of Consolidation (U)")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, mean_U, label="Mean U", color='blue')
plt.fill_between(time, mean_U - std_U, mean_U + std_U, color='blue', alpha=0.3, label="±1 Std Dev for U")
plt.plot(time, mean_Unavfac, label="Mean Unavfac", color='red', linestyle='--')
plt.fill_between(time, mean_Unavfac - std_Unavfac, mean_Unavfac + std_Unavfac, color='red', alpha=0.3, label="±1 Std Dev for Unavfac")
plt.title("Comparison of U and Unavfac (With Variability)")
plt.xlabel("Time (years)")
plt.ylabel("Average Degree of Consolidation")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, mean_U, label="Mean U", color='blue')
plt.plot(time, mean_Unavfac, label="Mean Unavfac", color='red', linestyle='--')
plt.title("Comparison of Mean U and Mean Unavfac Over Time")
plt.xlabel("Time (years)")
plt.ylabel("Average Degree of Consolidation")
plt.legend()
plt.grid()
plt.show()

percentiles_U = np.percentile(bottom_U_results, [25, 50, 75], axis=0)
percentiles_Unavfac = np.percentile(Unavfac_results, [25, 50, 75], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(time, percentiles_U[1], label="Median U", color='blue')
plt.plot(time, percentiles_Unavfac[1], label="Median Unavfac", color='red', linestyle='--')
plt.fill_between(time, percentiles_U[0], percentiles_U[2], color='blue', alpha=0.3, label="5th-95th Percentile for U")
plt.fill_between(time, percentiles_Unavfac[0], percentiles_Unavfac[2], color='red', alpha=0.3, label="5th-95th Percentile for Unavfac")
plt.title("Comparison of U and Unavfac Percentiles")
plt.xlabel("Time (years)")
plt.ylabel("Degree of Consolidation")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
for i in range(3):  # Plot 3 random simulations
    tmp = np.random.randint(i, num_simulations)
    plt.plot(time, bottom_U_results[tmp, :], label=f"Sim {tmp} U", linestyle='-')
    plt.plot(time, Unavfac_results[tmp, :], label=f"Sim {tmp} Unavfac", linestyle='--')

plt.title("Comparison of Individual Simulations for U and Unavfac")
plt.xlabel("Time (years)")
plt.ylabel("Degree of Consolidation")
plt.legend()
plt.grid()
plt.show()

# Select two random simulations
sim1, sim2 = np.random.choice(bottom_U_results.shape[0], size=2, replace=False)

# Get data for the selected simulations
U_sim1 = bottom_U_results[sim1, :]
U_sim2 = bottom_U_results[sim2, :]
Unavfac_sim1 = Unavfac_results[sim1, :]
Unavfac_sim2 = Unavfac_results[sim2, :]

# Create correlation plots
plt.figure(figsize=(12, 6))

# Plot for the first simulation
plt.subplot(1, 2, 1)
plt.scatter(U_sim1, Unavfac_sim1, color='blue', alpha=0.7, edgecolor='k')
plt.title(f"Correlation: Simulation {sim1}")
plt.xlabel("U (Bottom Results)")
plt.ylabel("Unavfac Results")
plt.grid()

# Plot for the second simulation
plt.subplot(1, 2, 2)
plt.scatter(U_sim2, Unavfac_sim2, color='red', alpha=0.7, edgecolor='k')
plt.title(f"Correlation: Simulation {sim2}")
plt.xlabel("U (Bottom Results)")
plt.ylabel("Unavfac Results")
plt.grid()

plt.tight_layout()
plt.show()

max_length = max(len(cv) for cv in cv_results)
cv_padded = np.full((num_simulations, max_length), np.nan)
for i, cv in enumerate(cv_results):
    cv_padded[i, :len(cv)] = cv

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(np.ma.masked_invalid(cv_padded), cmap='plasma', cbar=True, vmax=100, xticklabels='auto', yticklabels='auto')
plt.title("Heatmap of cv Values Across All Simulations and Elements")
plt.xlabel("Element Index")
plt.ylabel("Simulation Index")
plt.show()
