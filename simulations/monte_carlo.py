import numpy as np
from .option import Option

class MonteCarloSimulator:
    def __init__(self, option, M, dt):
        """
        Classe pour simuler le prix des options via Monte Carlo.
        :param option: Instance de la classe Option.
        :param M: Nombre de pas temporels.
        :param dt: Taille d'un pas temporel (T / M).
        """
        self.option = option
        self.M = M
        self.dt = dt

    def basic(self, I):
        """
        Méthode de Monte Carlo basique pour estimer le prix de l'option.
        :param I: Nombre de simulations.
        :return: Dictionnaire avec le prix estimé, la variance et l'écart type.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)
        return self._compute_results(hT, I)

    def antithetic(self, I):
        """
        Méthode Monte Carlo avec variables antithétiques pour améliorer la précision.
        :param I: Nombre de simulations.
        :return: Dictionnaire avec le prix estimé, la variance et l'écart type.
        """
        S, P = self._generate_paths_antithetic(I)
        hT = (self.option.payoff_call_asian(S) + self.option.payoff_call_asian(P)) / 2
        return self._compute_results(hT, I)

    def control_variate_geom(self, I, return_control = False):
        """
        Méthode Monte Carlo avec variates de contrôle géométriques.
        :param I: Nombre de simulations.
        :return: Dictionnaire avec le prix estimé, la variance et l'écart type.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)
        hT_geom = self.option.payoff_call_geometric(S)
        Z_control = hT_geom - np.mean(hT_geom)
        beta_opt = -np.cov(hT, Z_control, ddof=1)[0, 1] / np.var(Z_control, ddof=1)
        hT_controlled = hT + beta_opt * Z_control
        if return_control:
            return hT, Z_control
        else:
            return self._compute_results(hT, I)
    
    def control_variate(self, I, return_control=False):
        """
        Méthode Monte Carlo avec variates de contrôle arithmétiques.
        :param I: Nombre de simulations.
        :param return_control: Si True, retourne également les variates de contrôle.
        :return: Résultats du pricing ou (payoffs, variates de contrôle) si return_control=True.
        """
        S = self._generate_paths(I)
        hT = self.option.payoff_call_asian(S)  # Calcul du payoff asiatique arithmétique
        avg = np.mean(S[1:], axis=0)  # Moyenne arithmétique des prix
        Z_control = avg - np.mean(avg)  # Variate de contrôle basé sur la moyenne

        # Calcul du beta optimal
        beta_opt = -np.cov(hT, Z_control, ddof=1)[0, 1] / np.var(Z_control, ddof=1)
        hT_controlled = hT + beta_opt * Z_control

        if return_control:
            return hT, Z_control  # Retourne les payoffs et les variates de contrôle
        else:
            return self._compute_results(hT_controlled, I)  # Retourne les résultats standard

    def _generate_paths(self, I):
        """
        Génère les trajectoires pour le sous-jacent.
        :param I: Nombre de simulations.
        :return: Matrice des trajectoires simulées.
        """
        S = np.zeros((self.M + 1, I))
        S[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt +
                                     self.option.sigma * np.sqrt(self.dt) * Z)
        return S

    def _generate_paths_antithetic(self, I):
        """
        Génère les trajectoires pour la méthode antithétique.
        :param I: Nombre de simulations.
        :return: Deux matrices des trajectoires simulées (Z et -Z).
        """
        S = np.zeros((self.M + 1, I))
        P = np.zeros((self.M + 1, I))
        S[0] = P[0] = self.option.S0
        for t in range(1, self.M + 1):
            Z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt +
                                     self.option.sigma * np.sqrt(self.dt) * Z)
            P[t] = P[t - 1] * np.exp((self.option.r - self.option.sigma**2 / 2) * self.dt -
                                     self.option.sigma * np.sqrt(self.dt) * Z)
        return S, P

    def _compute_results(self, hT, I):
        """
        Calcule le prix estimé, la variance et l'écart type des simulations.
        :param hT: Payoffs simulés.
        :param I: Nombre de simulations.
        :return: Dictionnaire avec le prix estimé, la variance et l'écart type.
        """
        price_estimate = np.exp(-self.option.r * self.option.T) * np.mean(hT)
        price_variance = np.exp(-2 * self.option.r * self.option.T) * np.var(hT) / I
        price_std = np.sqrt(price_variance)
        return {
            "price": price_estimate,
            "variance": price_variance,
            "std_dev": price_std
        }