import numpy as np
from scipy.stats import norm
from simulations.monte_carlo import MonteCarlo

class HedgingStrategy:
    def __init__(self, option, S_paths, M, I):
        '''
        - M: number of time step for hedging position refresh
        '''
        self.option = option
        self.S_paths = S_paths
        self.M = M
        self.I = I

    def compute_hedging_strategy(self):
        xi = self._compute_xi(self, self.S_paths)
    
    def _compute_xi(self):
        time_values = (self.S_paths.index - self.S_paths.index[0]).days
        time_values_years = time_values / 365.0

        S_integral = np.zeros_like(self.S_paths)
        for i in range(1, len(self.S_paths)):
            dt = time_values_years[i] - time_values_years[i - 1]
            S_integral[i] = S_integral[i - 1] + (self.S_paths[i] + self.S_paths[i - 1])/2 * dt
        
        return (S_integral / self.option.T - self.option.K) / self.S_paths
    
    def _F(self, t):
        S = np.zeros((self.M + 1, self.I))
        S[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt + self.option.sigma * np.sqrt(self.dt) * Z)
    
