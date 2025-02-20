# Pricing Asian Options Using Monte Carlo Methods

## Project Overview

This repository contains the implementation of **Monte Carlo (MC) and randomized Quasi-Monte Carlo (RQMC) methods** for pricing **Asian options**, with an application to **Brent Crude**. The project focuses on **variance reduction techniques** such as **antithetic variables** and **control variates** to improve the efficiency and accuracy of option pricing.

The approach is explained in the report **Pricing Asian Options.pdf**, which also compares and discusses results.

### Key Features:
- **Implementation of Monte Carlo (MC) and Quasi-Monte Carlo (RQMC) methods** for pricing Asian options.
- **Application of variance reduction techniques**: 
  - Antithetic variables
  - Control variates (using geometric Asian options)
- **Comparison of different pricing methods** based on accuracy, computational efficiency, and convergence speed.
- **Application to Brent Crude options**, using real market data.

---

## Project Structure
 
📜 `main.py`                    _# Main script to run simulations_  
📁 `simulations`  
│── 📜 `monte_carlo.py`         _# MC and RQMC simulation methods_  
│── 📜 `option.py`              _# Class for defining options_  
📁 `data`  
│── 📜 `loader.py`              _# Data loading and preprocessing_  
📁 `analysis`  
│── 📜 `tools.py`               _# Utility functions (comparisons, etc.)_  
📄 `Pricing Asian Options.pdf`  _# Report containing theoretical foundations, results, and comparisons_  
📄 `README.md`

## Prerequisites

Required libraries:
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

## Author

This project was developed by **Simon Duong**, student at **ESCP & ENSAE**.
