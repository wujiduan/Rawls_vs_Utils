import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
import copy
import os
from multiprocessing import Process


test = pd.read_csv('read_sipp/sipp_2014_wave_1.csv')
incomes = np.round(test['INCOME'].values)
bin_edges = [x * 10000 for x in [int(np.floor(min(incomes)/10000)), -1] + list(np.arange(0, 50, step=5)) + [50, int(np.ceil(max(incomes)/10000)+1)]]
counts, _ = np.histogram(incomes, bins=bin_edges)
initial_value = []
for i in range(len(bin_edges)-1):
    # use np.ceil to make the population more diversed and representative
    initial_value += list(np.random.randint(bin_edges[i]/1000, bin_edges[i+1]/1000, int(np.ceil(counts[i]/200))))
initial_value = np.array(initial_value, dtype='float64')
print("initial_values:", initial_value)

agent_num = len(initial_value)
print("agent num: ", agent_num)
max_time_horizon = 10000

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

'''  
This function compare the empirical finite-horizon performance between maximin policy v.s. maximax, f-maximax, fg-maximax.
'''
def one_compare(b, sigma2, i, j):
    
    
    hetero_decay_lbs = np.clip(np.random.normal(10, np.sqrt(sigma2), agent_num), 1, a_max=None)
    hetero_decay_ubs = np.clip(np.random.normal(hetero_decay_lbs * b, np.sqrt(hetero_decay_lbs * b * sigma2), agent_num), hetero_decay_lbs, a_max=None)
    
    
    hetero_return_lbs = np.clip(np.random.normal(10 * b * (agent_num + 1000), np.sqrt((agent_num + 1000) * sigma2 * b), agent_num), 1, np.inf)
    hetero_return_ubs = np.clip(np.random.normal(10 * b * (agent_num + 1000) + 40 * agent_num, np.sqrt(((agent_num + 1000) * b + 40 * agent_num) * sigma2), 
                                agent_num), hetero_return_lbs, a_max=None)


    maximin_agent_utilities = np.zeros(agent_num)
    fg_maximax_agent_utilities = np.zeros(agent_num)
    f_maximax_agent_utilities = np.zeros(agent_num)
    maximax_agent_utilities = np.zeros(agent_num)
    random_agent_utilities = np.zeros(agent_num)
    

    # noise range
    noise_bound = max(hetero_return_ubs) + max(hetero_decay_ubs)
        
    
    sample_num = 50
    maximin_util_sum = 0.
    fg_maximax_util_sum = 0.
    f_maximax_util_sum = 0.
    maximax_util_sum = 0.
    random_util_sum = 0.

    for _ in range(sample_num):
        maximin_agent_utilities = initial_value.copy()
        fg_maximax_agent_utilities = initial_value.copy()
        f_maximax_agent_utilities = initial_value.copy()
        maximax_agent_utilities = initial_value.copy()
        random_agent_utilities = initial_value.copy()

        # now we compare maximin and {fg-maximax, f-maximax, maximax, random}
        for t in range(max_time_horizon-1):

            # print("time step:", t)
            # shared symmetric noises in [-(return_ub+decay_ub), +(return_ub+decay_ub)]
            temp_noise = np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

            
            # "putting out fires" policy
            # compute the expected intervention return and the shock decays
            min_utility_ind = np.argmin(maximin_agent_utilities)
            maximin_decay_indicator = np.ones(agent_num)
            maximin_decay_indicator[min_utility_ind] = 0
            maximin_agent_utilities[min_utility_ind] += np.round(np.clip(return_alphas[min_utility_ind]*maximin_agent_utilities[min_utility_ind]+return_betas[min_utility_ind], hetero_return_lbs[min_utility_ind], hetero_return_ubs[min_utility_ind]))
            temp_decay = np.round(np.clip(decay_alphas*maximin_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            maximin_agent_utilities -= temp_decay * maximin_decay_indicator
            maximin_agent_utilities += temp_noise
                    

            # maximax policy
            maximax_agent_utilities += temp_noise
            max_utility_ind = np.argmax(maximax_agent_utilities)
            maximax_agent_utilities[max_utility_ind] += np.round(np.clip(return_alphas[max_utility_ind] * maximax_agent_utilities[max_utility_ind] + return_betas[max_utility_ind], hetero_return_lbs[max_utility_ind], hetero_return_ubs[max_utility_ind]))
            maximax_decay_indicator = np.ones(agent_num)
            maximax_decay_indicator[max_utility_ind] = 0
            temp_decay = np.round(np.clip(decay_alphas*maximax_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            maximax_agent_utilities -= temp_decay * maximax_decay_indicator
                
            
            # fg_maximax policy
            temp_return = np.round(np.clip(return_alphas*fg_maximax_agent_utilities+return_betas, hetero_return_lbs, hetero_return_ubs))
            temp_decay = np.round(np.clip(decay_alphas*fg_maximax_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            temp_fplusg = temp_return + temp_decay
            max_gain_ind = np.argmax(temp_fplusg)

            fg_maximax_decay_indicator = np.ones(agent_num)
            fg_maximax_decay_indicator[max_gain_ind] = 0
            fg_maximax_agent_utilities -= temp_decay * fg_maximax_decay_indicator
            fg_maximax_agent_utilities[max_gain_ind] += temp_return[max_gain_ind]
            fg_maximax_agent_utilities += temp_noise

            # f_maximax policy
            temp_return = np.round(np.clip(return_alphas*fg_maximax_agent_utilities+return_betas, hetero_return_lbs, hetero_return_ubs))
            temp_decay = np.round(np.clip(decay_alphas*fg_maximax_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            max_f_ind = np.argmax(temp_return)

            f_maximax_decay_indicator = np.ones(agent_num)
            f_maximax_decay_indicator[max_f_ind] = 0
            f_maximax_agent_utilities -= temp_decay * f_maximax_decay_indicator
            f_maximax_agent_utilities[max_f_ind] += temp_return[max_f_ind]
            f_maximax_agent_utilities += temp_noise

            # random policy
            temp_decay = np.round(np.clip(decay_alphas*random_agent_utilities+decay_betas, hetero_decay_lbs, hetero_decay_ubs))
            temp_rand_ind = np.random.randint(0, agent_num)
            random_decay_indicator = np.ones(agent_num)
            random_decay_indicator[temp_rand_ind] = 0
            random_agent_utilities -= temp_decay * random_decay_indicator
            random_agent_utilities[temp_rand_ind] += np.round(np.clip(return_alphas[temp_rand_ind] * random_agent_utilities[temp_rand_ind] + return_betas[temp_rand_ind], hetero_return_lbs[temp_rand_ind], hetero_return_ubs[temp_rand_ind]))
            random_agent_utilities += temp_noise
                        
        maximin_util_sum += sum(maximin_agent_utilities) / (sample_num * max_time_horizon)
        fg_maximax_util_sum += sum(fg_maximax_agent_utilities) / (sample_num * max_time_horizon)
        f_maximax_util_sum += sum(f_maximax_agent_utilities) / (sample_num * max_time_horizon)
        maximax_util_sum += sum(maximax_agent_utilities) / (sample_num * max_time_horizon)
        random_util_sum += sum(random_agent_utilities) / (sample_num * max_time_horizon)

    print("maximin:", maximin_util_sum, ", fg maximax:", fg_maximax_util_sum, "f maximax:", f_maximax_util_sum, "maximax:", maximax_util_sum, "random:", random_util_sum)

    if maximin_util_sum >= max([fg_maximax_util_sum, f_maximax_util_sum, maximax_util_sum, random_util_sum]):
        with open(results_folder + '/winning_record.txt', 'a') as file:
            file.write(str(1) + "," + str(i) + "," + str(j) + "\n")
        return 1
    else:
        with open(results_folder + '/winning_record.txt', 'a') as file:
            file.write(str(0) + "," + str(i) + "," + str(j) + "\n")
        return 0
    
b_lb = 1.25
b_ub = 10
sigma_lb = 0.25
sigma_ub = 1000
bs = np.linspace(b_lb, b_ub, 50)
sigma2s = np.linspace(sigma_lb, sigma_ub, 50)
cell_sample_num = 50

if __name__ == "__main__":

    procs = []
    for i in range(len(bs)):
        for j in range(len(sigma2s)):
            for _ in range(cell_sample_num):
                p = Process(target=one_compare, args=(bs[i], sigma2s[j], i, j))
                p.start()
                procs.append(p)
    for p in procs:
        p.join()

    print("Finished!")
