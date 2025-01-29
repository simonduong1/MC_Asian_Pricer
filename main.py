import time
import pandas as pd
import matplotlib.pyplot as plt
from data.loader import DataLoader
from simulations.option import Option
from simulations.monte_carlo import MonteCarlo, RQMC
from analysis.tools import AnalysisTools
from simulations.hedging import HedgingStrategy

# Parameters initialisation
ticker = 'BZ=F'
start_date = '2023-10-02'
start_date = pd.to_datetime(start_date)
T = 1.0
K = 80
M = 52
dt = T / M
I_values = [100, 1000, 5000, 10000, 50000, 100000, 250000, 750000]

# Loading data
data_loader = DataLoader(ticker, start_date)
df, S0, sigma, r, last_date_available = data_loader.load_data()
print(f'Initial price: {S0}, Volatility: {sigma}, Risk-free rate: {r}')

#Loading analysis tools
analysis_tools = AnalysisTools()

# Definition of simulators
option = Option(S0, K, T, r, sigma)
mc_simulator = MonteCarlo(option, M, dt)
rqmc_simulator = RQMC(option, M, dt)

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


MC_methods = {
        "MC Basic": mc_simulator.basic,
        "MC Antithetic": mc_simulator.antithetic,
        "MC Control Variate": mc_simulator.control_variate_geom,
    }

RQMC_methods = {
        "RQMC Basic": rqmc_simulator.basic,
        "RQMCAntithetic": rqmc_simulator.antithetic,
        "RQMCControl Variate": rqmc_simulator.control_variate_geom,
    }

S_real_path = df.loc[start_date:(start_date + pd.DateOffset(years=1)), 'Close']

hedger = HedgingStrategy(option, S_real_path)
print(hedger._compute_xi())