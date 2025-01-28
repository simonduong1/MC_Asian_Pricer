import numpy as np

class Option:
    def __init__(self, S0, K, T, r, sigma):
        """
        This class represents an option.
        - S0: initial price of the underlying asset.
        - K: option's strike price.
        - T: maturity (in years).
        - r: continuously compounded risk-free rate.
        - sigma: volatily of the underlying asset.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def payoff_call_asian(self, S_paths):
        """
        Computes the payoff of an Asian call option (arithmetic average)
        - S_paths: matrix of simulated paths of the underlying asset.
        It returns the payoff of the Asian call for each simulation.
        """
        average_price = np.mean(S_paths[1:], axis=0)
        return np.maximum(average_price - self.K, 0)

    def payoff_put_asian(self, S_paths):
        """
        Computes the payoff of an Asian put option (arithmetic average)
        - S_paths: matrix of simulated paths of the underlying asset.
        It returns the payoff of the Asian call for each simulation.
        """
        average_price = np.mean(S_paths[1:], axis=0)
        return np.maximum(self.K - average_price, 0)

    def payoff_call_geometric(self, S_paths):
        """
        Computes the payoff of an Asian call option (geometric average)
        - S_paths: matrix of simulated paths of the underlying asset.
        It returns the payoff of the Asian call for each simulation.
        """
        geometric_average = np.exp(np.mean(np.log(S_paths[1:]), axis=0))
        return np.maximum(geometric_average - self.K, 0)

    def payoff_put_geometric(self, S_paths):
        """
        Computes the payoff of an Asian put option (geometric average)
        - S_paths: matrix of simulated paths of the underlying asset.
        It returns the payoff of the Asian call for each simulation.
        """
        geometric_average = np.exp(np.mean(np.log(S_paths[1:]), axis=0))
        return np.maximum(self.K - geometric_average, 0)