# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:36:10 2024

@author: bgiet
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

# loading the data
boys_data = pd.read_csv(r"C:\Users\bgiet\OneDrive\Documents\GitHub\ConflictingForces\ConflictingForces\sensitive_files\ConflictingForcesSensitive\girls_data.csv", encoding='ISO-8859-2')

# preparing the data
boys_data['shifted_SDQ'] = boys_data['SDQ_hyper_PCG_W2'] + 0.5
boys_data['shifted_Conscientious'] = boys_data['Conscientious_W2_PCG'] + 0.5
boys_data['log_maths'] = np.log(boys_data['Maths_points'])
boys_data['log_english'] = np.log(boys_data['English_points'])
boys_data['log_cognition'] = np.log(boys_data['cognition_pc'])
boys_data['log_SDQ'] = np.log(boys_data['shifted_SDQ'])
boys_data['log_Conscientious'] = np.log(boys_data['shifted_Conscientious'])

# defining the translog production function
def translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12):
    return (np.log(A) + alpha * log_C + beta * log_N + 
            0.5 * gamma1 * log_C**2 + 
            0.5 * gamma2 * log_N**2 + 
            gamma12 * log_C * log_N)

# defining model functions for SDQ and TIPI, for both Maths and English
def model_function_sdq(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = boys_data['log_SDQ']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

def model_function_tipi(log_C, A, alpha, beta, gamma1, gamma2, gamma12):
    log_N = boys_data['log_Conscientious']
    return translog_production_interaction(log_C, log_N, A, alpha, beta, gamma1, gamma2, gamma12)

# extracting the data for fitting
log_C = boys_data['log_cognition']
log_maths = boys_data['log_maths']
log_english = boys_data['log_english']

# function to fit model and print results
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
    
    return popt

# fitting and print results for all models
popt_sdq_maths = fit_and_print_results(model_function_sdq, log_C, log_maths, "SDQ Maths")
popt_tipi_maths = fit_and_print_results(model_function_tipi, log_C, log_maths, "TIPI Maths")
popt_sdq_english = fit_and_print_results(model_function_sdq, log_C, log_english, "SDQ English")
popt_tipi_english = fit_and_print_results(model_function_tipi, log_C, log_english, "TIPI English")

# mean values
mean_C = np.mean(boys_data['cognition_pc'])
mean_N_sdq = np.mean(boys_data['shifted_SDQ'])
mean_N_tipi = np.mean(boys_data['shifted_Conscientious'])

# functions for calculating metrics
def mp_C(A, C, N, alpha, beta, gamma1, gamma2, gamma12):
    log_C = np.log(C)
    log_N = np.log(N)
    return A * np.exp(alpha * log_C + beta * log_N + 0.5 * gamma1 * log_C**2 + 0.5 * gamma2 * log_N**2 + gamma12 * log_C * log_N) * \
           (alpha + gamma1 * log_C + gamma12 * log_N) / C

def mp_N(A, C, N, alpha, beta, gamma1, gamma2, gamma12):
    log_C = np.log(C)
    log_N = np.log(N)
    return A * np.exp(alpha * log_C + beta * log_N + 0.5 * gamma1 * log_C**2 + 0.5 * gamma2 * log_N**2 + gamma12 * log_C * log_N) * \
           (beta + gamma2 * log_N + gamma12 * log_C) / N

def oe_C(C, N, alpha, gamma1, gamma12):
    return alpha + gamma1 * np.log(C) + gamma12 * np.log(N)

def oe_N(C, N, beta, gamma2, gamma12):
    return beta + gamma2 * np.log(N) + gamma12 * np.log(C)

def es(C, N, alpha, beta, gamma1, gamma2, gamma12):
    oe_C_value = oe_C(C, N, alpha, gamma1, gamma12)
    oe_N_value = oe_N(C, N, beta, gamma2, gamma12)
    return 2 - (gamma1 / oe_C_value) + (gamma12 / oe_N_value) + (gamma12 / oe_C_value) - (gamma2 / oe_N_value)

def mrts(C, N, alpha, beta, gamma1, gamma2, gamma12):
    oe_C_value = oe_C(C, N, alpha, gamma1, gamma12)
    oe_N_value = oe_N(C, N, beta, gamma2, gamma12)
    return (oe_C_value / oe_N_value) * (N / C)

def calculate_metrics(popt, mean_C, mean_N):
    A, alpha, beta, gamma1, gamma2, gamma12 = popt
    
    mp_C_value = mp_C(A, mean_C, mean_N, alpha, beta, gamma1, gamma2, gamma12)
    mp_N_value = mp_N(A, mean_C, mean_N, alpha, beta, gamma1, gamma2, gamma12)
    oe_C_value = oe_C(mean_C, mean_N, alpha, gamma1, gamma12)
    oe_N_value = oe_N(mean_C, mean_N, beta, gamma2, gamma12)
    es_value = es(mean_C, mean_N, alpha, beta, gamma1, gamma2, gamma12)
    mrts_value = mrts(mean_C, mean_N, alpha, beta, gamma1, gamma2, gamma12)
    
    return {
        'mp_C': mp_C_value,
        'mp_N': mp_N_value,
        'oe_C': oe_C_value,
        'oe_N': oe_N_value,
        'es': es_value,
        'mrts': mrts_value
    }

# metrics for each model
metrics_sdq_maths = calculate_metrics(popt_sdq_maths, mean_C, mean_N_sdq)
metrics_tipi_maths = calculate_metrics(popt_tipi_maths, mean_C, mean_N_tipi)
metrics_sdq_english = calculate_metrics(popt_sdq_english, mean_C, mean_N_sdq)
metrics_tipi_english = calculate_metrics(popt_tipi_english, mean_C, mean_N_tipi)

# print metrics
def print_metrics(metrics, model_name):
    print(f"\nMetrics for {model_name}:")
    print(f"Marginal Product of C: {metrics['mp_C']:.5f}")
    print(f"Marginal Product of N: {metrics['mp_N']:.5f}")
    print(f"Output Elasticity of C: {metrics['oe_C']:.5f}")
    print(f"Output Elasticity of N: {metrics['oe_N']:.5f}")
    print(f"Elasticity of Substitution: {metrics['es']:.5f}")
    print(f"Marginal Rate of Technical Substitution: {metrics['mrts']:.5f}")

# print metrics for each model
print_metrics(metrics_sdq_maths, "Maths SDQ")
print_metrics(metrics_tipi_maths, "Maths TIPI")
print_metrics(metrics_sdq_english, "English SDQ")
print_metrics(metrics_tipi_english, "English TIPI")