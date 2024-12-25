'''
In this document, we compute the percentage when maximin policy
tends to perform better than the maximax policy in both short-term/long-term.
For simplicity and efficiency, the number of budgets is set to 1.
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing

import pickle 
import copy
import os
from multiprocessing import Process



test = pd.read_csv('read_sipp/sipp_2014_wave_1.csv')
incomes = np.round(test['INCOME'].values)
bin_edges = [x * 10000 for x in [int(np.floor(min(incomes)/10000)), -1] + list(np.arange(0, 50, step=5)) + [50, int(np.ceil(max(incomes)/10000)+1)]]
counts, _ = np.histogram(incomes, bins=bin_edges)
initial_value = []
# by setting budget to 1, we group every 200 individuals as a group
budget_num = 1
for i in range(len(bin_edges)-1):
    # use np.ceil to make the population more diversed and representative
    # initial_value += list(np.random.randint(bin_edges[i]/1000, bin_edges[i+1]/1000, int(np.ceil(counts[i]/200*budget_num))))
    initial_value += list(np.random.randint(bin_edges[i]/1000, bin_edges[i+1]/1000, int(np.ceil(counts[i]/200))))


initial_value = np.array(initial_value, dtype='float64')
print("initial_values:", initial_value)

agent_num = len(initial_value)
print("agent num: ", agent_num)
short_term_time_horizon = 10
long_term_time_horizon = 6000

param_folder = "sipp_linear_final_params"
if not os.path.exists(param_folder):
    os.makedirs(param_folder)
shape_params_path = param_folder + '/sipp_linear_shape_params.txt'
results_folder = "heterogeneous_results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
homo_hetero_signature = results_folder + "/homo_hetero_sipp_linear_"


# we use the same set of parameters we use for the SIPP simulation under homogeneous assumption.
return_alphas = np.zeros(agent_num)
return_betas = np.zeros(agent_num)
decay_alphas = np.zeros(agent_num)
decay_betas = np.zeros(agent_num)

with open(shape_params_path, 'r') as file:

    counter = 0
    for line in file:
        if counter >= agent_num:
            break
        params = line.split(',')
        return_alphas[counter] = params[0]
        return_betas[counter] = params[1]
        decay_alphas[counter] = params[2]
        decay_betas[counter] = params[3]
        counter += 1

b_lb = 0.05
b_ub = 1
sigma_lb = 0.25
sigma_ub = 100
bs = np.linspace(b_lb, b_ub, 50)
sigma2s = np.linspace(sigma_lb, sigma_ub, 50)
cell_sample_num = 50

'''  
This function compare the empirical finite-horizon performance between maximin policy v.s. maximax, f-maximax, fg-maximax.
'''
def one_compare(b, sigma2, i, j, cs):
    
    print("one compare begins:", i, j)
    
    hetero_decay_lbs = np.clip(np.random.normal(60 * b, np.sqrt(b**2 * sigma2), agent_num), 1, a_max=None)
    hetero_decay_ubs = np.clip(np.random.normal(60, np.sqrt(sigma2), agent_num), hetero_decay_lbs, a_max=None)
    

    # homo_return_lb = homo_decay_ub * agent_num * 4 / budget_num
    # homo_return_ub = homo_return_lb + (homo_decay_lb+homo_decay_ub) * agent_num / (2*budget_num)
    
    hetero_return_lbs = np.clip(np.random.normal(60 * agent_num * 4 / budget_num, np.sqrt((agent_num * 4 / budget_num)**2 * sigma2), agent_num), 1, a_max=None)
    hetero_return_ubs = np.clip(np.random.normal(60 * agent_num * 4 / budget_num + 35 * agent_num / budget_num, np.sqrt(((agent_num * 4 / budget_num) + 7/12 * agent_num / budget_num)**2 * sigma2), 
                                agent_num), hetero_return_lbs, a_max=None)

    # hetero_return_lbs = np.clip(np.random.normal(60 * (agent_num + 1000), np.sqrt((agent_num + 1000)**2 * sigma2), agent_num), 1, a_max=None)
    # hetero_return_ubs = np.clip(np.random.normal(60 * (agent_num + 1000 + 2/3 * agent_num), np.sqrt(((agent_num + 1000) + 2/3 * agent_num)**2 * sigma2), 
    #                             agent_num), hetero_return_lbs, a_max=None)


    # record another possibility 
    test_zeta = sum([1 - x/(x+y) for x,y in zip(hetero_decay_ubs, hetero_return_lbs)]) / sum([1/(x+y) for x,y in zip(hetero_decay_ubs, hetero_return_lbs)])

    maximin_agent_utilities = np.zeros(agent_num)
    maximax_agent_utilities = np.zeros(agent_num)


    # noise range
    noise_bound = max(hetero_return_ubs) + max(hetero_decay_ubs)
        
    maximin_short_term_util_sum = 0.
    maximax_short_term_util_sum = 0.
    maximin_long_term_util_sum = 0.
    maximax_long_term_util_sum = 0.

    sample_num = 50
    for s in range(sample_num):
        np.random.seed(1000000 * cs + 10000 * i + 100 * j + s)
        maximin_agent_utilities = initial_value.copy()
        maximax_agent_utilities = initial_value.copy()
        
        # now we compare maximin and {fg-maximax, f-maximax, maximax, random}
        for t in range(long_term_time_horizon):

            # print("time step:", t)
            # symmetric noises in [-(return_ub+decay_ub), +(return_ub+decay_ub)]
            # temp_noise1 = np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

            
            # "putting out fires" policy
            # compute the expected intervention return and the shock decays
            min_utility_ind = np.argmin(maximin_agent_utilities)
            maximin_decay_indicator = np.ones(agent_num)
            maximin_decay_indicator[min_utility_ind] = 0
            maximin_agent_utilities[min_utility_ind] += np.round(np.clip(return_alphas[min_utility_ind]*maximin_agent_utilities[min_utility_ind]+return_betas[min_utility_ind], hetero_return_lbs[min_utility_ind], hetero_return_ubs[min_utility_ind]))
            temp_decay = np.round(np.clip(decay_alphas*maximin_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            maximin_agent_utilities -= temp_decay * maximin_decay_indicator
            maximin_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))
                    

            # maximax policy
            max_utility_ind = np.argmax(maximax_agent_utilities)
            maximax_agent_utilities[max_utility_ind] += np.round(np.clip(return_alphas[max_utility_ind] * maximax_agent_utilities[max_utility_ind] + return_betas[max_utility_ind], hetero_return_lbs[max_utility_ind], hetero_return_ubs[max_utility_ind]))
            maximax_decay_indicator = np.ones(agent_num)
            maximax_decay_indicator[max_utility_ind] = 0
            temp_decay = np.round(np.clip(decay_alphas*maximax_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            maximax_agent_utilities -= temp_decay * maximax_decay_indicator
            maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

            if t < short_term_time_horizon:
                maximin_short_term_util_sum += sum(maximin_agent_utilities) / (sample_num * short_term_time_horizon)
                maximax_short_term_util_sum += sum(maximax_agent_utilities) / (sample_num * short_term_time_horizon)
                        
        maximin_long_term_util_sum += sum(maximin_agent_utilities) / (sample_num * long_term_time_horizon)
        maximax_long_term_util_sum += sum(maximax_agent_utilities) / (sample_num * long_term_time_horizon)
        
    # print("maximin short term:", maximin_short_term_util_sum, "maximax short term:", maximax_short_term_util_sum)

    if maximin_short_term_util_sum >= maximax_short_term_util_sum:
        
        with open(results_folder + '/winning_record_pool_short_term.txt', 'a') as file:
                file.write(str(i) + "," + str(j) + "," + str(1) + "\n")
            
    else:
        with open(results_folder + '/winning_record_pool_short_term.txt', 'a') as file:
                file.write(str(i) + "," + str(j) + "," + str(0) + "\n")

    if maximin_long_term_util_sum >= maximax_long_term_util_sum:
        
        with open(results_folder + '/winning_record_pool_long_term.txt', 'a') as file:
                file.write(str(i) + "," + str(j) + "," + str(1) + "\n")
        
    else:

        with open(results_folder + '/winning_record_pool_long_term.txt', 'a') as file:
                file.write(str(i) + "," + str(j) + "," + str(0) + "\n")
        
        
    return 0
            
    


if __name__ == "__main__":

    
    short_term_record_filename = 'heterogeneous_results/winning_record_pool_short_term.txt'
    long_term_record_filename = 'heterogeneous_results/winning_record_pool_long_term.txt'

    # clean the files before writing it
    with open(short_term_record_filename, 'w') as file:
        pass
    with open(long_term_record_filename, 'w') as file:
        pass


    params = []
    
    for i in range(len(bs)):
        for j in range(len(sigma2s)):
            for cs in range(cell_sample_num):
                params.append((bs[i], sigma2s[j], i, j, cs))

    pool_size = multiprocessing.cpu_count()
    with multiprocessing.Pool(pool_size) as pool:
        results = pool.starmap(one_compare, params)

    print("Finished!")


    with open(long_term_record_filename, 'r') as file: 
        winning_percentage_long_term_survival = np.zeros((len(bs), len(sigma2s)))
        winning_percentage_long_term_nsurvival = np.zeros((len(bs), len(sigma2s)))
        for line in file:
            values = line.strip().split(',')
            print("debug values:", values)
            winning_percentage_long_term_survival[int(values[0]), int(values[1])] += float(values[2])
            

        winning_percentage_long_term_survival = winning_percentage_long_term_survival / cell_sample_num
        
        # Plot the heatmap
        plt.imshow(winning_percentage_long_term_survival, extent=[sigma2s[0], sigma2s[-1], bs[0], bs[-1]], vmin=0, vmax=1, origin='lower', aspect='auto')
        plt.colorbar()
        plt.xlabel(r"Coefficient $\sigma^2$", fontsize=15)
        plt.ylabel(r"Coefficient $b$", fontsize=15)
        # Set fewer ticks
        plt.xticks(np.linspace(sigma_lb, sigma_ub, 5))
        plt.yticks(np.linspace(b_lb, b_ub, 5))
        plt.savefig("heterogeneous_results/maximin_winning_percentage_long_term_survival.pdf", bbox_inches='tight')
        # plt.show()


    with open(short_term_record_filename, 'r') as file:
        winning_percentage_short_term_survival = np.zeros((len(bs), len(sigma2s)))
        
        for line in file:
            values = line.strip().split(',')
            winning_percentage_short_term_survival[int(values[0]), int(values[1])] += float(values[2])
            
        winning_percentage_short_term_survival = winning_percentage_short_term_survival / cell_sample_num
        
        # print(winning_percentage_short_term.shape)
        # Plot the heatmap
        plt.figure()
        plt.imshow(winning_percentage_short_term_survival, extent=[sigma2s[0], sigma2s[-1], bs[0], bs[-1]], vmin=0, vmax=1, origin='lower', aspect='auto')
        # plt.imshow(winning_percentage_short_term)
        # plt.imshow(winning_percentage, )
        plt.colorbar()
        plt.xlabel(r"Coefficient $\sigma^2$", fontsize=15)
        plt.ylabel(r"Coefficient $b$", fontsize=15)
        # Set fewer ticks
        plt.xticks(np.linspace(sigma_lb, sigma_ub, 5))
        plt.yticks(np.linspace(b_lb, b_ub, 5))
        plt.savefig("heterogeneous_results/maximin_winning_percentage_short_term_survival.pdf", bbox_inches='tight')
        # plt.show()
        
        
        




