import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from ..simulations.monte_carlo import MonteCarlo


class AnalysisTools:
    def __init__(self):
        """
        This class gathers all the functions used to make analyses on estimation methods, parameters, etc.
        """
        pass

    def compare_I_values(self, pricing_method, M, dt, I_values):
        """
        This method compares the price estimations, and the standard errors for various values of I.
        Compare les estimations de prix et l'écart type pour différentes valeurs de I.
        - pricing_method: pricing method used.
        - option: object of the class Option on which the pricing method is applied.
        - M: number of time steps.
        - dt: size of time steps.
        - I_values: values of I (number of simulations) to be compared.
        It returns a pandas' dataframe containing three columns (I, Estimated Price, Standard Deviation).
        """
        results = []

        for I in I_values:
            result = pricing_method(I)
            results.append({
                'I': I,
                'Estimated Price': round(result['price'], 4),
                'Standard Deviation': round(result['std_dev'], 4)
            })

        results_df = pd.DataFrame(results)
        return results_df

    def compare_control_variates(self, option, M, dt, I_values, num_subsamples=10):
        """
        This methods compares the arithmetic average and the Geometric Asian Call option as control variates, on three different criteria.
        - option: object of the class Option.
        - M: number of time steps.
        - dt: size of time steps.
        - I_values: different values of I (number of simulations).
        - num_subsamples: number of subsamples used to evaluate the robustness of beta values.
        It returns a pandas' dataframe containing the results (correlation, Betas' standard deviation, computing time) and it plots the results.
        """

        # Initialize results storage
        corrs_geom, corrs_avg = [], []
        std_betas_geom, std_betas_avg = [], []
        avg_times_geom, avg_times_avg = [], []

        mc_simulator = MonteCarlo(option, M, dt)

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
        This method compares several pricing method based on their precision, computing time and efficiency.
        - methods: dictionary containing the methods to be compared.
        - I_values: different values of I (number of simulations).
        It returns a pandas' dataframe containing the results (price, standard deviation, computing time, efficiency) and plots the results.
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
        Generates plots to compare control variates.
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
        Generates plots to compare pricing methods.
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