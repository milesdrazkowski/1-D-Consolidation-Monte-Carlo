# This code implements a Monte Carlo simulation framework to determine
# whether a numerical implementation is necessary. This code focuses on geometric
# spatial variability and its effects. This code provides a probablistic comparison
# of the NAVFAC and a generic numerical implementation.

import math
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def lognormal_params(median, cv):
    mu = np.log(median**2 / np.sqrt(cv**2 + 1))
    sigma = np.sqrt(np.log(cv**2 + 1))
    return mu, sigma

# Monte Carlo parameters
num_simulations = 10000  # Number of Monte Carlo iterations

INDICT = 0
rr = []

# Time step parameters
dt = 1 / 12
tf = 10
steps = int(tf / dt)
time = np.arange(steps) * dt  # Time array

u0 = 1000 # Initial Load

# Storage for results
bottom_U_results = np.zeros((num_simulations, steps))  # Degree of consolidation at the bottom node
Unavfac_results = np.zeros((num_simulations, steps))

# Log-normal distribution parameters for cv and L
cv_median = 15
cv_cv = 1.2
L_median = 5
L_cv = 0.9
cv_mu, cv_sigma = lognormal_params(cv_median, cv_cv)
L_mu, L_sigma = lognormal_params(L_median, L_cv)

# Monte Carlo loop
for sim in range(num_simulations):
    # Randomize parameters
    layers = np.random.randint(2, 5)  # Number of layers (e.g., 2 to 4 layers)
    cv = np.random.lognormal(cv_mu, cv_sigma, size=layers)  # Log-normal distribution for cv
    L = np.random.lognormal(L_mu, L_sigma, size=layers)  # Log-normal distribution for L
    n = np.random.randint(10, 50, size=layers)  # Elements per layer (10 to 50)

    H = sum(L)  # Total height
    I = L / n  # Element sizes
    nt = sum(n)  # Total number of elements

    Heff = L[0]
    for i in range(1,layers-1,1):
        Heff = Heff + (L[i] * math.sqrt(cv[i] / cv[0]))

    # Initialize matrices
    K = np.zeros((nt + 1, nt + 1))
    KT = np.zeros((nt + 1, nt + 1))
    RS = np.zeros((nt + 1, steps))
    u = np.zeros((nt + 1, steps))
    U = np.zeros((nt + 1, steps))
    u[:, 0] = u0
    U[:, 0] = u0

    # Precompute constants
    cv_I = cv / I
    I_dt = I / dt

    # Assemble global matrices
    for j in range(len(n)):
        count = n[j]
        for i in range(count):
            idx = i + sum(n[:j])
            K[idx:idx + 2, idx:idx + 2] += cv_I[j] * np.array([[1, -1], [-1, 1]])
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

    T = cv[-1]*time/((Heff/2)**2)
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

# Plot histogram of the degree of consolidation at the bottom node at final time
plt.figure(figsize=(8, 5))
plt.hist(bottom_U_results[:, -1], bins=20, alpha=0.75, color='blue', edgecolor='black')
plt.title("Histogram of Final Degree of Consolidation at Bottom Node")
plt.xlabel("Degree of Consolidation (U)")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Plot mean and variance of degree of consolidation over time
mean_U = np.mean(bottom_U_results, axis=0)
std_U = np.std(bottom_U_results, axis=0)

plt.figure(figsize=(8, 5))
plt.plot(time, mean_U, label="Average Degree of Consolidation")
plt.fill_between(time, mean_U - std_U, mean_U + std_U, color='blue', alpha=0.3, label="±1 Std Dev")
plt.title("Mean and Variance of Average Degree of Consolidation")
plt.xlabel("Time (years)")
plt.ylabel("Degree of Consolidation (U)")
plt.legend()
plt.grid()
plt.show()
