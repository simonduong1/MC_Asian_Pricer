import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import qmc, norm
from .option import Option

class Simulator(ABC):
    def __init__(self, option, M, dt):
        """
        Abstract class for simulators
        - option: object of the Option class.
        - M: number of time steps.
        - dt: size of time steps.
        """
        self.option = option
        self.M = M
        self.dt = dt

    @abstractmethod
    def _generate_paths(self, I):
        """
        Abstract method for generating simulation paths. It is overdriven in each derived class.
        """
        pass

    @abstractmethod
    def _generate_paths_antithetic(self, I):
        """
        Abstract method for generating simulation paths using antithetic variables. It is overdriven in each derived class.
        """
        pass

    def basic(self, I):
        """
        Basic MC method.
        - I: Number of simulations.
        It returns a dictionary with the estimated price, the variance, and the standard deviation.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)
        return self._compute_results(hT, I)

    def antithetic(self, I):
        """
        MC method with antithetic variables.
        - I: Number of simulations.
        It returns a dictionary with the estimated price, the variance, and the standard deviation.
        """
        S, P = self._generate_paths_antithetic(I)
        hT = (self.option.payoff_call_asian(S) + self.option.payoff_call_asian(P)) / 2
        return self._compute_results(hT, I)

    def control_variate_geom(self, I, return_control = False):
        """
        MC method with geometric asian call option as control variate.
        - I: Number of simulations.
        It returns:
        - (if return_control = False): a dictionary with the estimated price, the variance, and the standard deviation.
        - (if return_control = True): the vector of payoffs adjusted using the control variate, and the control variate itself.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)
        hT_geom = self.option.payoff_call_geometric(S)
        Z_control = hT_geom - np.mean(hT_geom)
        beta_opt = -np.cov(hT, Z_control, ddof=1)[0, 1] / np.var(Z_control, ddof=1)
        hT_controlled = hT + beta_opt * Z_control
        if return_control:
            return hT_controlled, Z_control
        else:
            return self._compute_results(hT_controlled, I)
    
    def control_variate(self, I, return_control=False):
        """
        MC method with arithmetic average as control variate.
        - I: Number of simulations.
        It returns:
        - (if return_control = False): a dictionary with the estimated price, the variance, and the standard deviation.
        - (if return_control = True): the vector of payoffs adjusted using the control variate, and the control variate itself.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)  # Calcul du payoff asiatique arithmétique
        avg = np.mean(S[1:], axis=0)  # Moyenne arithmétique des prix
        Z_control = avg - np.mean(avg)  # Variate de contrôle basé sur la moyenne

        # Calcul du beta optimal
        beta_opt = -np.cov(hT, Z_control, ddof=1)[0, 1] / np.var(Z_control, ddof=1)
        hT_controlled = hT + beta_opt * Z_control

        if return_control:
            return hT_controlled, Z_control  # Retourne les payoffs et les variates de contrôle
        else:
            return self._compute_results(hT_controlled, I)  # Retourne les résultats standard
    
    def compute_delta(self, S_path, I, t, epsilon=0.01):
        time_values = (S_path.index - S_path.index[0]).days / 366 #2024 is a leap year
        index = time_values.get_indexer([t], method="ffill")[0]
        if time_values[index] == t:
            index -= 1
        S_historic = S_path[:index+1]

    def _compute_results(self, hT, I):
        """
        Computes the estimated price, the variance, and the standard deviation of the simulations.
        - hT: simulated payoffs.
        - I: number of simulations.
        It returns a dictionary with the estimated price, the variance, and the standard deviation.
        """
        price_estimate = np.exp(-self.option.r * self.option.T) * np.mean(hT)
        price_variance = np.exp(-2 * self.option.r * self.option.T) * np.var(hT) / I
        price_std = np.sqrt(price_variance)
        return {
            "price": price_estimate,
            "variance": price_variance,
            "std_dev": price_std
        }

class MonteCarlo(Simulator):
    def _generate_paths(self, I):
        """
        Overdrives the function for object of the class MonteCarlo.
        Generate paths of the underlying asset.
        - I: Number of simulations.
        It returns a matrix containing the simulated paths.
        """
        S = np.zeros((self.M + 1, I))
        S[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt + self.option.sigma * np.sqrt(self.dt) * Z)
        return S

    def _generate_paths_antithetic(self, I):
        """
        Overdrives the function for object of the class MonteCarlo.
        Generate paths of the underlying asset for the antithetic variables method.
        - I: Number of simulations.
        It returns a matrix containing the simulated paths.
        """
        S = np.zeros((self.M + 1, I))
        P = np.zeros((self.M + 1, I))
        S[0] = P[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt + self.option.sigma * np.sqrt(self.dt) * Z)
            P[t] = P[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt - self.option.sigma * np.sqrt(self.dt) * Z)
        return S, P
    
class RQMC(Simulator):
    def _generate_paths(self, I):
        """
        Overdrives the function for object of the class RQMC.
        Generate paths of the underlying asset.
        - I: Number of simulations.
        It returns a matrix containing the simulated paths.
        """
        sampler = qmc.Sobol(self.M, scramble=True)
        quasi_uniform = sampler.random(I)
        quasi_normal = norm.ppf(np.clip(quasi_uniform, 1e-10, 1 - 1e-10))

        S = np.zeros((self.M + 1, I))
        S[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = quasi_normal[:, t-1]
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt + self.option.sigma * np.sqrt(self.dt) * Z)
        return S

    def _generate_paths_antithetic(self, I):
        """
        Overdrives the function for object of the class RQMC.
        Generate paths of the underlying asset for the antithetic variables method.
        - I: Number of simulations.
        It returns a matrix containing the simulated paths.
        """
        sampler = qmc.Sobol(self.M, scramble=True)
        quasi_uniform = sampler.random(I)
        quasi_normal = norm.ppf(np.clip(quasi_uniform, 1e-10, 1 - 1e-10))

        S = np.zeros((self.M + 1, I))
        P = np.zeros((self.M + 1, I))
        S[0] = P[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = quasi_normal[:, t-1]
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt + self.option.sigma * np.sqrt(self.dt) * Z)
            P[t] = P[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt - self.option.sigma * np.sqrt(self.dt) * Z)
        return S, P