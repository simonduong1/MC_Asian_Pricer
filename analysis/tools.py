import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


class AnalysisTools:
    def __init__(self):
        pass

    def compare_I_values(self, pricing_method, option, M, dt, I_values):
        """
        Compare les estimations de prix et l'écart type pour différentes valeurs de I.
        :param pricing_method: Méthode de pricing à utiliser (fonction).
        :param option: Instance de la classe Option.
        :param M: Nombre de pas temporels.
        :param dt: Taille d'un pas temporel.
        :param I_values: Liste des nombres de simulations.
        :return: DataFrame des résultats.
        """
        results = []

        for I in I_values:
            result = pricing_method(I)
            results.append({
                'I': I,
                'Estimated Price': result['price'],
                'Standard Deviation': result['std_dev']
            })

        results_df = pd.DataFrame(results)
        return results_df

    def compare_control_variates(self, option, M, dt, I_values, num_subsamples=10):
        """
        Compare l'efficacité des variates de contrôle géométriques et arithmétiques.
        :param option: Instance de la classe Option.
        :param M: Nombre de pas temporels.
        :param dt: Taille d'un pas temporel.
        :param I_values: Liste des nombres de simulations.
        :param num_subsamples: Nombre de sous-échantillons.
        :return: DataFrame des résultats.
        """
        from simulations.monte_carlo import MonteCarloSimulator

        # Initialize results storage
        corrs_geom, corrs_avg = [], []
        std_betas_geom, std_betas_avg = [], []
        avg_times_geom, avg_times_avg = [], []

        mc_simulator = MonteCarloSimulator(option, M, dt)

        for I in I_values:
            sample_size = I // num_subsamples
            betas_geom, betas_avg = [], []
            times_geom, times_avg = [], []

            for _ in range(num_subsamples):
                I_sample = sample_size

                # Geometric control variate
                start_time = time.time()
                hT_geom, Z_control_geom = mc_simulator.control_variate_geom(I_sample, return_control=True)
                times_geom.append(time.time() - start_time)

                # Arithmetic control variate
                start_time = time.time()
                hT_avg, Z_control_avg = mc_simulator.control_variate(I_sample, return_control=True)
                times_avg.append(time.time() - start_time)

                # Calculate betas for each subsample
                cov_geom = np.cov(hT_geom, Z_control_geom, ddof=1)[0, 1]
                var_Z_geom = np.var(Z_control_geom, ddof=1)
                beta_opt_geom = -cov_geom / var_Z_geom
                betas_geom.append(beta_opt_geom)

                cov_avg = np.cov(hT_avg, Z_control_avg, ddof=1)[0, 1]
                var_Z_avg = np.var(Z_control_avg, ddof=1)
                beta_opt_avg = -cov_avg / var_Z_avg
                betas_avg.append(beta_opt_avg)

            avg_times_geom.append(np.mean(times_geom))
            avg_times_avg.append(np.mean(times_avg))
            std_betas_geom.append(np.std(betas_geom, ddof=1))
            std_betas_avg.append(np.std(betas_avg, ddof=1))

            # Full correlations
            full_hT_geom, full_Z_control_geom = mc_simulator.control_variate_geom(I, return_control=True)
            full_hT_avg, full_Z_control_avg = mc_simulator.control_variate(I, return_control=True)
            corrs_geom.append(np.corrcoef(full_hT_geom, full_Z_control_geom)[0, 1])
            corrs_avg.append(np.corrcoef(full_hT_avg, full_Z_control_avg)[0, 1])

        # Compile results into a DataFrame
        results_df = pd.DataFrame({
            "I": I_values,
            "Geom Correlation": corrs_geom,
            "Avg Correlation": corrs_avg,
            "Geom Beta StdDev": std_betas_geom,
            "Avg Beta StdDev": std_betas_avg,
            "Geom Avg Time (s)": avg_times_geom,
            "Avg Avg Time (s)": avg_times_avg
        })

        self._plot_control_variates(I_values, results_df)
        return results_df

    def compare_methods(self, methods, I_values):
        """
        Compare plusieurs méthodes de pricing selon leur précision, temps de calcul et efficacité.
        :param methods: Dictionnaire des méthodes à comparer.
        :param I_values: Liste des nombres de simulations.
        :return: DataFrame des résultats.
        """
        results = {method: {"Prices": [], "StdDev": [], "Times": [], "Efficiency": []} for method in methods}

        for method_name, method_func in methods.items():
            for I in I_values:
                start_time = time.time()
                result = method_func(I)
                computing_time = time.time() - start_time
                results[method_name]["Prices"].append(result["price"])
                results[method_name]["StdDev"].append(result["std_dev"])
                results[method_name]["Times"].append(computing_time)
                results[method_name]["Efficiency"].append(1 / (result["variance"] * computing_time))

        df = pd.DataFrame({"I": I_values})
        for method, data in results.items():
            df[f"{method} Price"] = data["Prices"]
            df[f"{method} StdDev"] = data["StdDev"]
            df[f"{method} Time"] = data["Times"]
            df[f"{method} Efficiency"] = data["Efficiency"]

        self._plot_methods_comparison(I_values, results)
        return df

    def _plot_control_variates(self, I_values, results_df):
        """
        Génère des graphiques pour les variates de contrôle.
        """

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Correlation plot
        axes[0].plot(I_values, results_df["Geom Correlation"], marker='o', label="Geometric Asian Control", color='blue', markersize=4, linewidth=0.7)
        axes[0].plot(I_values, results_df["Avg Correlation"], marker='o', label="Average Control", color='black', markersize=4, linewidth=0.7)
        axes[0].set_xscale('log')
        axes[0].set_title(r"\textbf{Correlation with Payoff}", fontsize=12)
        axes[0].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[0].set_ylabel(r"\textbf{Correlation}", fontsize=10)
        axes[0].tick_params(axis='both', which='major', labelsize=9)
        axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[0].legend(loc="best", fontsize=8, frameon=False)

        # Beta StdDev plot
        axes[1].plot(I_values, results_df["Geom Beta StdDev"], marker='o', label="Geometric Asian Control", color='blue', markersize=4, linewidth=0.7)
        axes[1].plot(I_values, results_df["Avg Beta StdDev"], marker='o', label="Average Control", color='black', markersize=4, linewidth=0.7)
        axes[1].set_xscale('log')
        axes[1].set_title(r"\textbf{Standard Deviation of Optimal Beta}", fontsize=12)
        axes[1].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[1].set_ylabel(r"\textbf{Standard Deviation}", fontsize=10)
        axes[1].tick_params(axis='both', which='major', labelsize=9)
        axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[1].legend(loc="best", fontsize=8, frameon=False)

        # Computation time plot
        axes[2].plot(I_values, results_df["Geom Avg Time (s)"], marker='o', label="Geometric Asian Control", color='blue', markersize=4, linewidth=0.7)
        axes[2].plot(I_values, results_df["Avg Avg Time (s)"], marker='o', label="Average Control", color='black', markersize=4, linewidth=0.7)
        axes[2].set_xscale('log')
        axes[2].set_title(r"\textbf{Computation Time}", fontsize=12)
        axes[2].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[2].set_ylabel(r"\textbf{Time (s)}", fontsize=10)
        axes[2].tick_params(axis='both', which='major', labelsize=9)
        axes[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[2].legend(loc="best", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.show()

    def _plot_methods_comparison(self, I_values, results):
        """
        Génère des graphiques pour comparer les méthodes.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

        # Standard deviation
        for method, data in results.items():
            axes[0, 0].plot(I_values, data["StdDev"], marker='o', label=method, markersize=4, linewidth=0.7)
        axes[0,0].set_xscale('log')
        axes[0,0].set_title(r"\textbf{Precision (Standard Deviation)}", fontsize=12)
        axes[0,0].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[0,0].set_ylabel(r"\textbf{Standard Deviation}", fontsize=10)
        axes[0,0].tick_params(axis='both', which='major', labelsize=9)
        axes[0,0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[0,0].legend(loc="best", fontsize=8, frameon=False)

        # Estimated price
        for method, data in results.items():
            axes[0, 1].plot(I_values, data["Prices"], marker='o', label=method,  markersize=4, linewidth=0.7)
        axes[0,1].set_xscale('log')
        axes[0,1].set_title(r"\textbf{Convergence (Estimated Prices)}", fontsize=12)
        axes[0,1].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[0,1].set_ylabel(r"\textbf{Estimated Price}", fontsize=10)
        axes[0,1].tick_params(axis='both', which='major', labelsize=9)
        axes[0,1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[0,1].legend(loc="best", fontsize=8, frameon=False)

        # Computation time
        for method, data in results.items():
            axes[1, 0].plot(I_values, data["Times"], marker='o', label=method,  markersize=4, linewidth=0.7)
        axes[1,0].set_xscale('log')
        axes[1,0].set_title(r"\textbf{Computation Time}", fontsize=12)
        axes[1,0].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[1,0].set_ylabel(r"\textbf{Computation Time (s)}", fontsize=10)
        axes[1,0].tick_params(axis='both', which='major', labelsize=9)
        axes[1,0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[1,0].legend(loc="best", fontsize=8, frameon=False)

        # Efficiency
        for method, data in results.items():
            axes[1, 1].plot(I_values, data["Efficiency"], marker='o', label=method,  markersize=4, linewidth=0.7)
        axes[1,1].set_xscale('log')
        axes[1,1].set_title(r"\textbf{Efficiency}", fontsize=12)
        axes[1,1].set_xlabel(r"\textbf{Number of Simulations (I)}", fontsize=10)
        axes[1,1].set_ylabel(r"\textbf{Efficiency}", fontsize=10)
        axes[1,1].tick_params(axis='both', which='major', labelsize=9)
        axes[1,1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[1,1].legend(loc="best", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.show()