# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:16:36 2024

@author: bgiet
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

# Load the data
girls_data = pd.read_csv(r"C:\Users\bgiet\OneDrive\Documents\GitHub\ConflictingForces\ConflictingForces\sensitive_files\ConflictingForcesSensitive\girls_data.csv", encoding='ISO-8859-2')

# Prepare the data
girls_data['shifted_SDQ'] = girls_data['SDQ_hyper_PCG_W2'] + 0.5
girls_data['shifted_Conscientious'] = girls_data['Conscientious_W2_PCG'] + 0.5
girls_data['log_maths'] = np.log(girls_data['Maths_points'])
girls_data['log_english'] = np.log(girls_data['English_points'])
girls_data['log_cognition'] = np.log(girls_data['cognition_pc'])
girls_data['log_SDQ'] = np.log(girls_data['shifted_SDQ'])
girls_data['log_Conscientious'] = np.log(girls_data['shifted_Conscientious'])

# Define the translog production function
def translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12):
    return (np.log(A) + alpha * log_C + beta * log_N + 
            0.5 * gamma1 * log_C**2 + 
            0.5 * gamma2 * log_N**2 + 
            gamma12 * log_C * log_N)

# Define model functions for SDQ and TIPI, for both Maths and English
def model_function_sdq_maths(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = girls_data['log_SDQ']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

def model_function_tipi_maths(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = girls_data['log_Conscientious']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

def model_function_sdq_english(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = girls_data['log_SDQ']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

def model_function_tipi_english(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = girls_data['log_Conscientious']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

# Extract the data for fitting
log_C = girls_data['log_cognition']
log_maths = girls_data['log_maths']
log_english = girls_data['log_english']

# Function to fit model and print results
def fit_and_print_results(model_func, log_C, log_Y, model_name):
    popt, pcov = curve_fit(model_func, log_C, log_Y, 
                           p0=[1, 0.5, 0.5, 0.01, 0.01, 0.01],
                           bounds=([0.1, -1, -1, -1, -1, -1], [10, 2, 2, 1, 1, 1]))
    
    perr = np.sqrt(np.diag(pcov))
    p_values = 2 * (1 - norm.cdf(np.abs(popt / perr)))
    ci = 1.96 * perr

    print(f"\nResults for {model_name} model:")
    for param, value, err, p_value, ci_val in zip(['A', 'alpha', 'beta', 'gamma1', 'gamma2', 'gamma12'], 
                                                  popt, perr, p_values, ci):
        print(f"{param} = {value:.5f} ± {err:.5f}, p-value = {p_value:.5f}, 95% CI: [{value-ci_val:.5f}, {value+ci_val:.5f}]")

# Fit and print results for all models
fit_and_print_results(model_function_sdq_maths, log_C, log_maths, "SDQ Maths")
fit_and_print_results(model_function_tipi_maths, log_C, log_maths, "TIPI Maths")
fit_and_print_results(model_function_sdq_english, log_C, log_english, "SDQ English")
fit_and_print_results(model_function_tipi_english, log_C, log_english, "TIPI English")