import numpy as np

class Option:
    def __init__(self, S0, K, T, r, sigma):
        """
        Classe représentant une option.
        :param S0: Prix initial de l'actif sous-jacent.
        :param K: Strike de l'option.
        :param T: Maturité (en années).
        :param r: Taux sans risque (continu).
        :param sigma: Volatilité de l'actif sous-jacent.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def payoff_call_asian(self, S_paths):
        """
        Calcule le payoff pour un call asiatique (moyenne arithmétique).
        :param S_paths: Matrice des trajectoires simulées de l'actif sous-jacent.
        :return: Payoff du call asiatique pour chaque simulation.
        """
        average_price = np.mean(S_paths[1:], axis=0)
        return np.maximum(average_price - self.K, 0)

    def payoff_put_asian(self, S_paths):
        """
        Calcule le payoff pour un put asiatique (moyenne arithmétique).
        :param S_paths: Matrice des trajectoires simulées de l'actif sous-jacent.
        :return: Payoff du put asiatique pour chaque simulation.
        """
        average_price = np.mean(S_paths[1:], axis=0)
        return np.maximum(self.K - average_price, 0)

    def payoff_call_geometric(self, S_paths):
        """
        Calcule le payoff pour un call asiatique géométrique.
        :param S_paths: Matrice des trajectoires simulées de l'actif sous-jacent.
        :return: Payoff du call asiatique géométrique pour chaque simulation.
        """
        geometric_average = np.exp(np.mean(np.log(S_paths[1:]), axis=0))
        return np.maximum(geometric_average - self.K, 0)

    def payoff_put_geometric(self, S_paths):
        """
        Calcule le payoff pour un put asiatique géométrique.
        :param S_paths: Matrice des trajectoires simulées de l'actif sous-jacent.
        :return: Payoff du put asiatique géométrique pour chaque simulation.
        """
        geometric_average = np.exp(np.mean(np.log(S_paths[1:]), axis=0))
        return np.maximum(self.K - geometric_average, 0)