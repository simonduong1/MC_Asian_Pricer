# Pricing Asian Options Using Monte Carlo Methods

## Project Overview

This repository contains the implementation of **Monte Carlo (MC) and randomized Quasi-Monte Carlo (RQMC) methods** for pricing **Asian options**, with an application to **Brent Crude**. The project focuses on **variance reduction techniques** such as **antithetic variables** and **control variates** to improve the efficiency and accuracy of option pricing.

The approach is explained is the report **Pricing Asian Options.pdf**, which also compares and discusses results.

### Key Features:
- **Implementation of Monte Carlo (MC) and Quasi-Monte Carlo (RQMC) methods** for pricing Asian options.
- **Application of variance reduction techniques**: 
  - Antithetic variables
  - Control variates (using geometric Asian options)
- **Comparison of different pricing methods** based on accuracy, computational efficiency, and convergence speed.
- **Application to Brent Crude options**, using real market data.

---

## Project Structure

📁 MC_Asian_Pricer  
│── 📜 main.py                    # Main script to run simulations  
📁 simulations  
    │── 📜 monte_carlo.py         # MC and RQMC simulation methods  
    │── 📜 option.py              # Class for defining options  
📁 data  
    │── 📜 loader.py              # Data loading and preprocessing  
📁 analysis  
    │── 📜 tools.py               # Utility functions (comparisons, etc.)  
│── 📄 Pricing Asian Options.pdf  # Report containing theoretical foundations, results, and comparisons  
│── 📄 README.md

## Prerequisites

Required libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
