import time
import pandas as pd
import matplotlib.pyplot as plt
from data.loader import DataLoader
from simulations.option import Option
from simulations.monte_carlo import MonteCarloSimulator
'''from simulations.quasi_monte_carlo import QuasiMonteCarloSimulator'''
from analysis.tools import AnalysisTools

# Initialisation des paramètres
ticker = 'BZ=F'
start_date = '2023-10-02'
start_date = pd.to_datetime(start_date)
T = 1.0
K = 80
M = 52
dt = T / M
I_values = [100, 1000, 5000, 10000, 50000, 100000, 250000, 750000]

# Chargement des données
data_loader = DataLoader(ticker, start_date)
df, S0, sigma, r, last_date_available = data_loader.load_data()
print(f'Initial price: {S0}, Volatility: {sigma}, Risk-free rate: {r}')

#Loading analysis tools
analysis_tools = AnalysisTools()

# Définition des simulateurs
option = Option(S0, K, T, r, sigma)
mc_simulator = MonteCarloSimulator(option, M, dt)
'''qmc_simulator = QuasiMonteCarloSimulator(S0, K, T, r, sigma, M, dt)'''

#Graphs settings
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})


methods = {
        "MC Basic": mc_simulator.basic,
        "Antithetic": mc_simulator.antithetic,
        "Control Variate": mc_simulator.control_variate_geom,
    }

results_df = analysis_tools.compare_methods(methods, I_values)
print(results_df)