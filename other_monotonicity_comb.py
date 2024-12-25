import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing

import pickle 
import copy
import os
from multiprocessing import Process
from utility_funcs import compute_gini
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import heapq


'''  
We define the framework for comparison, and use return/decay functions as input parameters. 
This framework defined the return & decay is varying w.r.t. the utility, and they all include some symmetric noises.
The codes are vectorized w.r.t. agent_num.
'''
def run_comparison_fast(sample_num, signature, agent_num, return_ubs, return_lbs, decay_ubs, decay_lbs, return_alphas, 
                        return_betas, decay_alphas, decay_betas, max_time_horizon, initial_value, policies, if_gini, budget_num):
    
    if "maximin" or "all" in policies:
        maximin_agent_utilities = np.zeros(agent_num)
        maximin_mean_utilities = np.zeros((max_time_horizon, sample_num))
        maximin_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        maximin_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        if if_gini:
            maximin_gini = np.zeros((max_time_horizon, sample_num))


    if "fg_maximax" or "all" in policies:
        fg_maximax_agent_utilities = np.zeros(agent_num)
        fg_maximax_mean_utilities = np.zeros((max_time_horizon, sample_num))
        fg_maximax_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        fg_maximax_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        if if_gini:
            fg_maximax_gini = np.zeros((max_time_horizon, sample_num))


    if "f_maximax" or "all" in policies:
            
        f_maximax_agent_utilities = np.zeros(agent_num)
        f_maximax_mean_utilities = np.zeros((max_time_horizon, sample_num))
        f_maximax_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        f_maximax_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        if if_gini:
            f_maximax_gini = np.zeros((max_time_horizon, sample_num))

    if "g_maximax" or "all" in policies:
            
        g_maximax_agent_utilities = np.zeros(agent_num)
        g_maximax_mean_utilities = np.zeros((max_time_horizon, sample_num))
        g_maximax_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        g_maximax_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        if if_gini:
            g_maximax_gini = np.zeros((max_time_horizon, sample_num))


    if "maximax" or "all" in policies:
        maximax_agent_utilities = np.zeros(agent_num)
        maximax_mean_utilities = np.zeros((max_time_horizon, sample_num))
        maximax_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        maximax_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        if if_gini:
            maximax_gini = np.zeros((max_time_horizon, sample_num))


    if "random" or "all" in policies:
        random_agent_utilities = np.zeros(agent_num)
        # mean utility at each time step
        random_mean_utilities = np.zeros((max_time_horizon, sample_num))
        random_mean_growth_rate = np.zeros((max_time_horizon-1, sample_num))
        # save whole trajectories to validate asymptotic behavior
        # this is for recording the behavior of all agents
        random_whole_utilities = np.zeros((agent_num, max_time_horizon, sample_num))
        # record gini coefficient
        if if_gini:
            random_gini = np.zeros((max_time_horizon, sample_num))

    # noise range
    noise_bound = max(return_ubs) + max(decay_ubs)


    for s in range(sample_num):

        print("sample id:", s)
        maximin_agent_utilities = initial_value.copy()
        fg_maximax_agent_utilities = initial_value.copy()
        f_maximax_agent_utilities = initial_value.copy()
        # for max-g policy, we use a slightly different tie-breaker and hence need to sort the values
        g_maximax_agent_utilities = initial_value.copy()
        # comment the following line for testing different tie-breaker
        g_maximax_agent_utilities = sorted(g_maximax_agent_utilities)
        maximax_agent_utilities = initial_value.copy()
        random_agent_utilities = initial_value.copy()


        maximin_mean_utilities[0, s] = np.mean(maximin_agent_utilities)
        if if_gini:
            _, maximin_gini[0, s] = compute_gini(maximin_agent_utilities, 1/(2*agent_num))
        maximin_whole_utilities[:, 0, s] = maximin_agent_utilities.copy()


        maximax_mean_utilities[0, s] = np.mean(maximax_agent_utilities) 
        if if_gini:
            _, maximax_gini[0, s] = compute_gini(maximax_agent_utilities, 1/(2*agent_num))
        maximax_whole_utilities[:, 0, s] = maximax_agent_utilities.copy()


        fg_maximax_mean_utilities[0, s] = np.mean(fg_maximax_agent_utilities)
        if if_gini:
            _, fg_maximax_gini[0, s] = compute_gini(fg_maximax_agent_utilities, 1/(2*agent_num))
        fg_maximax_whole_utilities[:, 0, s] = fg_maximax_agent_utilities.copy()


        f_maximax_mean_utilities[0, s] = np.mean(f_maximax_agent_utilities)
        if if_gini:
            _, f_maximax_gini[0, s] = compute_gini(f_maximax_agent_utilities, 1/(2*agent_num))
        f_maximax_whole_utilities[:, 0, s] = f_maximax_agent_utilities.copy()

        g_maximax_mean_utilities[0, s] = np.mean(g_maximax_agent_utilities)
        if if_gini:
            _, g_maximax_gini[0, s] = compute_gini(g_maximax_agent_utilities, 1/(2*agent_num))
        g_maximax_whole_utilities[:, 0, s] = g_maximax_agent_utilities.copy()


        random_mean_utilities[0, s] = np.mean(random_agent_utilities)
        if if_gini:
            _, random_gini[0, s] = compute_gini(random_agent_utilities, 1/(2*agent_num))
        random_whole_utilities[:, 0, s] = random_agent_utilities.copy()

        # now we compare two policies: putting out fires and greedy one 
        for t in range(max_time_horizon-1):
            

            print("sample id:", s, ", time step:", t)
            # shared symmetric noises in [-(return_ub+decay_ub), +(return_ub+decay_ub)]
            # comment the following line to use different noises for each policies
            # temp_noise = np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

            if "maximin" or "all" in policies:
            # "putting out fires" policy
            # compute the expected intervention return and the shock decays
            
                min_indices = heapq.nsmallest(budget_num, range(len(maximin_agent_utilities)), key=maximin_agent_utilities.__getitem__)
                # min_utility_ind = np.argmin(maximin_agent_utilities)
                maximin_decay_indicator = np.ones(agent_num)
                for ind in min_indices: 
                    maximin_decay_indicator[ind] = 0
                temp_return = np.round(np.clip(return_alphas*maximin_agent_utilities+return_betas, return_lbs, return_ubs))
                maximin_agent_utilities += temp_return * (1-maximin_decay_indicator)
                temp_decay = np.round(np.clip(decay_alphas*maximin_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                maximin_agent_utilities -= temp_decay * maximin_decay_indicator
                # add noises
                maximin_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))
                    
                maximin_mean_utilities[t+1, s] = np.mean(maximin_agent_utilities)
                maximin_mean_growth_rate[t, s] = (maximin_mean_utilities[t+1, s]-maximin_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, maximin_gini[t+1, s] = compute_gini(maximin_agent_utilities, 1/(2*agent_num))
                maximin_whole_utilities[:, t+1, s] = maximin_agent_utilities.copy()

            if "maximax" or "all" in policies:
                # maximax policy
                # add noises
                maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))
                max_indices = heapq.nlargest(budget_num, range(len(maximax_agent_utilities)), key=maximax_agent_utilities.__getitem__)

                
                maximax_decay_indicator = np.ones(agent_num)
                for ind in max_indices:
                    maximax_decay_indicator[ind] = 0
                temp_return = np.round(np.clip(return_alphas * maximax_agent_utilities + return_betas, return_lbs, return_ubs))
                maximax_agent_utilities += temp_return * (1-maximax_decay_indicator)
                temp_decay = np.round(np.clip(decay_alphas*maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                maximax_agent_utilities -= temp_decay * maximax_decay_indicator
                

                maximax_mean_utilities[t+1, s] = np.mean(maximax_agent_utilities)
                maximax_mean_growth_rate[t, s] = (maximax_mean_utilities[t+1, s]-maximax_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, maximax_gini[t+1, s] = compute_gini(maximax_agent_utilities, 1/(2*agent_num))
                maximax_whole_utilities[:, t+1, s] = maximax_agent_utilities.copy()

            if "fg_maximax" or "all" in policies:
                # fg_maximax policy
                temp_return = np.round(np.clip(return_alphas*fg_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
                temp_decay = np.round(np.clip(decay_alphas*fg_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                temp_fplusg = temp_return + temp_decay
                max_fg_indices = heapq.nlargest(budget_num, range(len(temp_fplusg)), key=temp_fplusg.__getitem__)

                # max_gain_ind = np.argmax(temp_fplusg)

                fg_maximax_decay_indicator = np.ones(agent_num)
                for ind in max_fg_indices:
                    fg_maximax_decay_indicator[ind] = 0
                fg_maximax_agent_utilities -= temp_decay * fg_maximax_decay_indicator
                fg_maximax_agent_utilities += temp_return * (1-fg_maximax_decay_indicator)
                # add noises
                fg_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

                fg_maximax_mean_utilities[t+1, s] = np.mean(fg_maximax_agent_utilities)
                fg_maximax_mean_growth_rate[t, s] = (fg_maximax_mean_utilities[t+1, s]-fg_maximax_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, fg_maximax_gini[t+1, s] = compute_gini(fg_maximax_agent_utilities, 1/(2*agent_num))
                fg_maximax_whole_utilities[:, t+1, s] = fg_maximax_agent_utilities.copy()


            if "f_maximax" or "all" in policies:
                # f_maximax policy
                temp_return = np.round(np.clip(return_alphas*f_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
                temp_decay = np.round(np.clip(decay_alphas*f_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                max_f_indices = heapq.nlargest(budget_num, range(len(f_maximax_agent_utilities)), key=f_maximax_agent_utilities.__getitem__)

                # max_f_ind = np.argmax(temp_return)

                f_maximax_decay_indicator = np.ones(agent_num)
                for ind in max_f_indices:
                    f_maximax_decay_indicator[ind] = 0
                f_maximax_agent_utilities -= temp_decay * f_maximax_decay_indicator
                f_maximax_agent_utilities += temp_return * (1-f_maximax_decay_indicator)
                # add noises
                f_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

                f_maximax_mean_utilities[t+1, s] = np.mean(f_maximax_agent_utilities)
                f_maximax_mean_growth_rate[t, s] = (f_maximax_mean_utilities[t+1, s]-f_maximax_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, f_maximax_gini[t+1, s] = compute_gini(f_maximax_agent_utilities, 1/(2*agent_num))
                f_maximax_whole_utilities[:, t+1, s] = f_maximax_agent_utilities.copy()


            if "g_maximax" or "all" in policies:
                # g_maximax policy
                temp_return = np.round(np.clip(return_alphas*g_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
                temp_decay = np.round(np.clip(decay_alphas*g_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                # since welfare values have been sorted in ascending order, returning the smallest index is enough for obtaining
                # the individual with the highest g with smallest welfare level
                
                max_g_indices = heapq.nlargest(budget_num, range(len(temp_decay)), key=temp_decay.__getitem__)
                g_maximax_decay_indicator = np.ones(agent_num)
                for ind in max_g_indices:
                    g_maximax_decay_indicator[ind] = 0
                g_maximax_agent_utilities -= temp_decay * g_maximax_decay_indicator
                g_maximax_agent_utilities += temp_return * (1-g_maximax_decay_indicator)
                # add noises
                g_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))
                # comment the following line to use min-index tie-breaking rule for g-max policy.
                g_maximax_agent_utilities = sorted(g_maximax_agent_utilities)
                g_maximax_mean_utilities[t+1, s] = np.mean(g_maximax_agent_utilities)
                g_maximax_mean_growth_rate[t, s] = (g_maximax_mean_utilities[t+1, s]-g_maximax_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, g_maximax_gini[t+1, s] = compute_gini(g_maximax_agent_utilities, 1/(2*agent_num))
                g_maximax_whole_utilities[:, t+1, s] = g_maximax_agent_utilities.copy()



            if "random" or "all" in policies:
                # random policy
                temp_decay = np.round(np.clip(decay_alphas*random_agent_utilities+decay_betas, decay_lbs, decay_ubs))
                temp_rand_indices = np.random.randint(0, agent_num, budget_num)
                random_decay_indicator = np.ones(agent_num)
                for ind in temp_rand_indices:
                    random_decay_indicator[ind] = 0
                random_agent_utilities -= temp_decay * random_decay_indicator
                temp_return = np.round(np.clip(return_alphas * random_agent_utilities + return_betas, return_lbs, return_ubs))
                random_agent_utilities += temp_return * (1-random_decay_indicator)
                # add noises
                random_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))
                            
                random_mean_utilities[t+1, s] = np.mean(random_agent_utilities)
                random_mean_growth_rate[t, s] = (random_mean_utilities[t+1, s]-random_mean_utilities[0, s]) / (t+1)
                if if_gini:
                    _, random_gini[t+1, s] = compute_gini(random_agent_utilities, 1/(2*agent_num))
                random_whole_utilities[:, t+1, s] = random_agent_utilities.copy()

        # We take the measure of utilitarianism: average utility, \sum_{i=1}^N Ui(t)/(N*t)
    if "maximin" or "all" in policies:

        maximin_mean_growth_rate_mean_discrete = np.mean(maximin_mean_growth_rate, axis=1)
        maximin_mean_growth_rate_std_discrete = np.std(maximin_mean_growth_rate, axis=1)
        
        with open(signature+"maximin_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(maximin_mean_growth_rate_mean_discrete, file)
        with open(signature+"maximin_mean_growth_rate_std_discrete.pk", 'wb') as file:
                pickle.dump(maximin_mean_growth_rate_std_discrete, file)
        with open(signature+"maximin_whole_utilities.pk", 'wb') as file:
            pickle.dump(maximin_whole_utilities, file)


        if if_gini:
            maximin_gini_mean_discrete = np.mean(maximin_gini, axis=1)
            maximin_gini_std_discrete = np.std(maximin_gini, axis=1)
            
            with open(signature+"maximin_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(maximin_gini_mean_discrete, file)
            with open(signature+"maximin_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(maximin_gini_std_discrete, file)


    if "maximax" or "all" in policies:
        maximax_mean_growth_rate_mean_discrete = np.mean(maximax_mean_growth_rate, axis=1)
        maximax_mean_growth_rate_std_discrete = np.std(maximax_mean_growth_rate, axis=1)
        
        with open(signature+"maximax_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(maximax_mean_growth_rate_mean_discrete, file)
        with open(signature+"maximax_mean_growth_rate_std_discrete.pk", 'wb') as file:
            pickle.dump(maximax_mean_growth_rate_std_discrete, file)
        with open(signature+"maximax_whole_utilities.pk", 'wb') as file:
            pickle.dump(maximax_whole_utilities, file)

        if if_gini:
            maximax_gini_mean_discrete = np.mean(maximax_gini, axis=1)
            maximax_gini_std_discrete = np.std(maximax_gini, axis=1)

            with open(signature+"maximax_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(maximax_gini_mean_discrete, file)
            with open(signature+"maximax_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(maximax_gini_std_discrete, file)



    if "fg_maximax" or "all" in policies:

        fg_maximax_mean_growth_rate_mean_discrete = np.mean(fg_maximax_mean_growth_rate, axis=1)
        fg_maximax_mean_growth_rate_std_discrete = np.std(fg_maximax_mean_growth_rate, axis=1)

        with open(signature+"fg_maximax_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(fg_maximax_mean_growth_rate_mean_discrete, file)
        with open(signature+"fg_maximax_mean_growth_rate_std_discrete.pk", 'wb') as file:
            pickle.dump(fg_maximax_mean_growth_rate_std_discrete, file)
        with open(signature+"fg_maximax_whole_utilities.pk", 'wb') as file:
            pickle.dump(fg_maximax_whole_utilities, file)


        if if_gini:
            fg_maximax_gini_mean_discrete = np.mean(fg_maximax_gini, axis=1)
            fg_maximax_gini_std_discrete = np.std(fg_maximax_gini, axis=1)

            with open(signature+"fg_maximax_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(fg_maximax_gini_mean_discrete, file)
            with open(signature+"fg_maximax_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(fg_maximax_gini_std_discrete, file)



    if "f_maximax" or "all" in policies:
        f_maximax_mean_growth_rate_mean_discrete = np.mean(f_maximax_mean_growth_rate, axis=1)
        f_maximax_mean_growth_rate_std_discrete = np.std(f_maximax_mean_growth_rate, axis=1)

        with open(signature+"f_maximax_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(f_maximax_mean_growth_rate_mean_discrete, file)
        with open(signature+"f_maximax_mean_growth_rate_std_discrete.pk", 'wb') as file:
            pickle.dump(f_maximax_mean_growth_rate_std_discrete, file)
        with open(signature+"f_maximax_whole_utilities.pk", 'wb') as file:
            pickle.dump(f_maximax_whole_utilities, file)


        if if_gini:
            f_maximax_gini_mean_discrete = np.mean(f_maximax_gini, axis=1)
            f_maximax_gini_std_discrete = np.std(f_maximax_gini, axis=1)

            with open(signature+"f_maximax_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(f_maximax_gini_mean_discrete, file)
            with open(signature+"f_maximax_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(f_maximax_gini_std_discrete, file)

    if "g_maximax" or "all" in policies:
        g_maximax_mean_growth_rate_mean_discrete = np.mean(g_maximax_mean_growth_rate, axis=1)
        g_maximax_mean_growth_rate_std_discrete = np.std(g_maximax_mean_growth_rate, axis=1)

        with open(signature+"g_maximax_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(g_maximax_mean_growth_rate_mean_discrete, file)
        with open(signature+"g_maximax_mean_growth_rate_std_discrete.pk", 'wb') as file:
            pickle.dump(g_maximax_mean_growth_rate_std_discrete, file)
        with open(signature+"g_maximax_whole_utilities.pk", 'wb') as file:
            pickle.dump(g_maximax_whole_utilities, file)


        if if_gini:
            g_maximax_gini_mean_discrete = np.mean(g_maximax_gini, axis=1)
            g_maximax_gini_std_discrete = np.std(g_maximax_gini, axis=1)

            with open(signature+"g_maximax_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(g_maximax_gini_mean_discrete, file)
            with open(signature+"g_maximax_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(g_maximax_gini_std_discrete, file)

        

    if "random" or "all" in policies:
        
        random_mean_growth_rate_mean_discrete = np.mean(random_mean_growth_rate, axis=1)
        random_mean_growth_rate_std_discrete = np.std(random_mean_growth_rate, axis=1)

        with open(signature+"random_mean_growth_rate_mean_discrete.pk", 'wb') as file:
            pickle.dump(random_mean_growth_rate_mean_discrete, file)
        with open(signature+"random_mean_growth_rate_std_discrete.pk", 'wb') as file:
            pickle.dump(random_mean_growth_rate_std_discrete, file)
        with open(signature+"random_whole_utilities.pk", 'wb') as file:
            pickle.dump(random_whole_utilities, file)

        if if_gini:
            random_gini_mean_discrete = np.mean(random_gini, axis=1)
            random_gini_std_discrete = np.std(random_gini, axis=1)

            with open(signature+"random_gini_mean_discrete.pk", 'wb') as file:
                pickle.dump(random_gini_mean_discrete, file)
            with open(signature+"random_gini_std_discrete.pk", 'wb') as file:
                pickle.dump(random_gini_std_discrete, file)
        

    return 0



test = pd.read_csv('read_sipp/sipp_2014_wave_1.csv')
incomes = np.round(test['INCOME'].values)
print("population size:", len(incomes))
print("max income:", max(incomes)) # 1140011
print("min income:", min(incomes)) # -369981
bin_edges = [x * 10000 for x in [int(np.floor(min(incomes)/10000)), -1] + list(np.arange(0, 50, step=5)) + [50, int(np.ceil(max(incomes)/10000)+1)]]
print(bin_edges)
print('bin num', len(bin_edges)-1)
plt.hist(incomes, bins = bin_edges, range=(int(np.floor(min(incomes)/10000))*10000, int(np.ceil(max(incomes)/10000)+1)*10000))
plt.show()
counts, _ = np.histogram(incomes, bins=bin_edges)
print(type(counts))
print(counts)
initial_value = []
# budget num is set to 1
for i in range(len(bin_edges)-1):
    # use np.ceil to make the population more diversed and representative
    initial_value += list(np.random.randint(bin_edges[i]/1000, bin_edges[i+1]/1000, int(np.ceil(counts[i]/200))))
initial_value = np.array(initial_value, dtype='float64')
print("initial_values:", initial_value)

# agent_num = len(incomes)
agent_num = len(initial_value)
print("agent num:", agent_num)
homo_decay_ub = 60
homo_decay_lb = 10
# f^- > (N-1)*g^+ ensures the positivity
homo_return_lb = homo_decay_ub * (agent_num + 1000)
homo_return_ub = homo_return_lb + 40 * agent_num
print("return upperbound:", homo_return_ub)
print("return lowerbound:", homo_return_lb)



f_monotonicities = ['increasing', 'decreasing', 'constant']
g_monotonicities = ['increasing', 'decreasing', 'constant']
param_folder = "sipp_linear_final_params"
if not os.path.exists(param_folder):
    os.makedirs(param_folder)
results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
homo_hetero_signature = results_folder + "/homo_hetero_sipp_linear_" 

# 6000, 100
max_time_horizon = 6000
sample_num = 100
# possible policies: "maximin", "maximax", "fg_maximax", "f_maximax", "random"
policies = ["all"]
if_gini = False
budget_num = 1
# set adjust_plot to False to generate data, to True to only ajust the plot
adjust_plot = True

if not adjust_plot:

    ''' 
    generate and record shape parameters for different monotonicity combination.

    '''
    for f_mono in f_monotonicities:
        for g_mono in g_monotonicities:
            shape_params_path = param_folder + '/sipp_linear_shape_params_' + f_mono + "_" + g_mono + '.txt'
            
            try: 
                # 'a' mode - to append
                # 'x' mode - generate only if not exists
                # 'w' mode - generate a new set of params
                with open(shape_params_path, 'w') as file:
                    
                    counter = 0
                    while counter < agent_num:

                        # we fix two end points and interpolate two linear functions.
                        # as long as return_ub-return_lb > decay_ub-decay_lb
                        # left_turning_x = np.random.randint(-2800, -2500)
                        if f_mono == 'increasing':
                            
                            return_left_turning_x = np.random.randint(-2100, -2000)
                            return_right_turning_x = np.random.randint(400, 800)
                            
                            temp_return_alpha = (homo_return_ub - homo_return_lb) / (return_right_turning_x - return_left_turning_x)
                            temp_return_beta = homo_return_lb - return_left_turning_x * temp_return_alpha

                            if g_mono == 'decreasing':       
                            
                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_lb - homo_decay_ub) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_ub - decay_left_turning_x * temp_decay_alpha

                            elif g_mono == 'increasing':

                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_ub - homo_decay_lb) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_lb - decay_left_turning_x * temp_decay_alpha

                            # g_mono == 'constant'
                            else:
                                temp_decay_alpha = 0.
                                temp_decay_beta = homo_decay_lb


                        elif f_mono == 'decreasing':
                            
                            return_left_turning_x = np.random.randint(-2100, -2000)
                            return_right_turning_x = np.random.randint(400, 800)
                            
                            temp_return_alpha = (homo_return_lb - homo_return_ub) / (return_right_turning_x - return_left_turning_x)
                            temp_return_beta = homo_return_ub - return_left_turning_x * temp_return_alpha

                            
                            if g_mono == 'increasing':
                                
                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_ub - homo_decay_lb) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_lb - decay_left_turning_x * temp_decay_alpha

                            elif g_mono == 'decreasing':
                                
                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_lb - homo_decay_ub) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_ub - decay_left_turning_x * temp_decay_alpha

                            # g_mono == 'constant'
                            else:
                                
                                temp_decay_alpha = 0.
                                temp_decay_beta = homo_decay_lb

                            
                        # f_mono == 'constant'
                        else:

                            temp_return_alpha = 0.
                            temp_return_beta = homo_return_ub

                            if g_mono == 'increasing':

                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_ub - homo_decay_lb) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_lb - decay_left_turning_x * temp_decay_alpha

                            elif g_mono == 'decreasing':

                                decay_left_turning_x = np.random.randint(-1900, -1850)
                                decay_right_turning_x = np.random.randint(-1600, -1500)

                                temp_decay_alpha = (homo_decay_lb - homo_decay_ub) / (decay_right_turning_x - decay_left_turning_x)
                                temp_decay_beta = homo_decay_ub - decay_left_turning_x * temp_decay_alpha

                            # g_mono == 'constant'
                            else:
                                temp_decay_alpha = 0.
                                temp_decay_beta = homo_decay_lb

                        file.write(str(temp_return_alpha) + "," + str(temp_return_beta) + "," + str(temp_decay_alpha) + "," + str(temp_decay_beta) + "\n")
                        counter += 1
                        print(counter)

                        
                        
            except FileExistsError:
                print("Shape parameters already generated!")


    
    for f_mono in f_monotonicities:
        for g_mono in g_monotonicities:
            
            temp_shape_param_path = param_folder + '/sipp_linear_shape_params_' + f_mono + "_" + g_mono + '.txt'
            homo_return_alphas = np.zeros(agent_num)
            homo_return_betas = np.zeros(agent_num)
            homo_decay_alphas = np.zeros(agent_num)
            homo_decay_betas = np.zeros(agent_num)

            with open(temp_shape_param_path, 'r') as file:

                counter = 0
                for line in file:
                    # if the param size and the agent number doesn't match, probably 
                    # you need to regenerate the shape parameters
                    params = line.split(',')
                    homo_return_alphas[counter] = params[0]
                    homo_return_betas[counter] = params[1]
                    homo_decay_alphas[counter] = params[2]
                    homo_decay_betas[counter] = params[3]
                    counter += 1

            homo_return_ubs = np.array([homo_return_ub for _ in range(agent_num)])
            homo_return_lbs = np.array([homo_return_lb for _ in range(agent_num)])
            homo_decay_ubs = np.array([homo_decay_ub for _ in range(agent_num)])
            homo_decay_lbs = np.array([homo_decay_lb for _ in range(agent_num)])


            temp_signature = homo_hetero_signature + "_" + f_mono + "_" + g_mono + "_"
            run_comparison_fast(sample_num, temp_signature, agent_num, return_ubs=homo_return_ubs, return_lbs=homo_return_lbs, decay_ubs=homo_decay_ubs, decay_lbs=homo_decay_lbs,
                        return_alphas=homo_return_alphas, return_betas=homo_return_betas, decay_alphas=homo_decay_alphas, decay_betas=homo_decay_betas, max_time_horizon=max_time_horizon, initial_value=initial_value, policies=policies, if_gini=if_gini, budget_num=1)


# expected zeta for min-U policy and expected zeta for max-U policy, the expected behavior for other 
# policies is still uncertain
# expected zeta varies for different monotonicities.
# for f_mono in [increasing, decreasing, constant] and g_mono in [increasing, decreasing, constant]
# when it's constant, we assume the return is homo_return_ub and the decay is homo_decay_lb
homo_expected_zeta = [[(homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num], 
                      [(homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num], 
                      [(homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num]]
print(homo_expected_zeta)
homo_expected_zeta_greedy = [[(homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num],
                             [(homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_lb*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num],
                             [(homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_ub) / agent_num, (homo_return_ub*budget_num - (agent_num-budget_num) * homo_decay_lb) / agent_num]]
print("expected utilitarian rbar:", homo_expected_zeta_greedy)


# how to plot 9 plots that share the same legend.
xs = np.array([k for k in range(max_time_horizon-1)])
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
y_ulims = [[385., 385., 388.], [350., 380., 385.], [385., 385., 415.]]
y_llims = [[295., 330., 340.], [290., 290., 335.], [330., 330., 375.]]
# match the order of the table
f_perturbed_ind = [1, 0, 2]
g_perturbed_ind = [1, 0, 2]
for i in range(3):
    for j in range(3):
        
        print("plot:", i, j)
        f_mono = f_monotonicities[f_perturbed_ind[i]]
        g_mono = g_monotonicities[g_perturbed_ind[j]]
        temp_expected_rawlsian_zeta = homo_expected_zeta[f_perturbed_ind[i]][g_perturbed_ind[j]]
        temp_expected_greedy_zeta = homo_expected_zeta_greedy[f_perturbed_ind[i]][g_perturbed_ind[j]]
        temp_signature = homo_hetero_signature + "_" + f_mono + "_" + g_mono + "_"

        axes[j, i].set_xlabel("Time step t", fontsize=18)
        axes[j, i].set_ylabel("Social welfare", fontsize=18)
        if "g_maximax" or "all" in policies:
            with open(temp_signature+"g_maximax_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                g_maximax_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"g_maximax_mean_growth_rate_std_discrete.pk", 'rb') as file:
                g_maximax_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(g_maximax_mean_growth_rate_mean_discrete, label = "max-g (Rawlsian)")
            axes[j, i].fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

        if "maximin" or "all" in policies:

            with open(temp_signature+"maximin_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                maximin_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"maximin_mean_growth_rate_std_discrete.pk", 'rb') as file:
                maximin_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(maximin_mean_growth_rate_mean_discrete, label="min-U (Rawlsian)")
            axes[j, i].fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


        if "fg_maximax" or "all" in policies:

            with open(temp_signature+"fg_maximax_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                fg_maximax_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"fg_maximax_mean_growth_rate_std_discrete.pk", 'rb') as file:
                fg_maximax_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(fg_maximax_mean_growth_rate_mean_discrete, label = "max-fg (utilitarian)")
            axes[j, i].fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)



        if "f_maximax" or "all" in policies:
            with open(temp_signature+"f_maximax_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                f_maximax_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"f_maximax_mean_growth_rate_std_discrete.pk", 'rb') as file:
                f_maximax_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(f_maximax_mean_growth_rate_mean_discrete, label = "max-f (utilitarian)")
            axes[j, i].fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)



        if "maximax" or "all" in policies:
            with open(temp_signature+"maximax_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                maximax_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"maximax_mean_growth_rate_std_discrete.pk", 'rb') as file:
                maximax_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(maximax_mean_growth_rate_mean_discrete, label="max-U (utilitarian)")
            axes[j, i].fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



        if "random" or "all" in policies:

            with open(temp_signature+"random_mean_growth_rate_mean_discrete.pk", 'rb') as file:
                random_mean_growth_rate_mean_discrete = pickle.load(file)
            with open(temp_signature+"random_mean_growth_rate_std_discrete.pk", 'rb') as file:
                random_mean_growth_rate_std_discrete = pickle.load(file)

            axes[j, i].plot(random_mean_growth_rate_mean_discrete, label="Random")
            axes[j, i].fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)



        # plot hlines
        hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
        hline_labels = [r'$\bar{R}_{min-U}$ (Rawlsian)', r'$\bar{R}_{max-U}$ (utilitarian)']
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_ind = [1, 4]
        for l in range(len(hlines)):
            axes[j, i].hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]], label=hline_labels[l])

        # plt.xlim(0, 1000)
        axes[j, i].set_ylim(y_llims[f_perturbed_ind[i]][g_perturbed_ind[j]], y_ulims[f_perturbed_ind[i]][g_perturbed_ind[j]])
        
        axes[j, i].grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        
        # add zoom-in camera to the plot
        if i==0 and j==0:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 700, 800, 347, 350
            axins = inset_axes(axes[0, 0], width="20%", height="20%", loc="upper center")
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[0, 0].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[0, 0], axins, loc1=2, loc2=4, fc="none", ec='0.5')

            # zoom in lower overlapped curves
            # zoom in upper overlapped curves
            x12, x22, y12, y22 = 300, 400, 310, 315
            axins2 = inset_axes(axes[0, 0], width="20%", height="20%", loc="lower center", bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=axes[0, 0].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins2.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins2.plot(maximin_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins2.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins2.plot(maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins2.plot(random_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins2.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes[0, 0].indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes[0, 0], axins2, loc1=2, loc2=4, fc="none", ec='0.5')

        if i==1 and j==0:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 950, 1050, 374, 377
            axins = inset_axes(axes[0, 1], width="20%", height="20%", loc="center", bbox_to_anchor=(-0.05, 0.2, 0.8, 0.8), bbox_transform=axes[0, 1].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[0, 1].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[0, 1], axins, loc1=1, loc2=3, fc="none", ec='0.5')

            # zoom in lower overlapped curves
            x12, x22, y12, y22 = 950, 1050, 341, 343
            axins2 = inset_axes(axes[0, 1], width="20%", height="20%", loc="lower center", bbox_to_anchor=(-0.05, 0.2, 0.8, 0.8), bbox_transform=axes[0, 1].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins2.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins2.plot(maximin_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins2.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins2.plot(maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins2.plot(random_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins2.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes[0, 1].indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes[0, 1], axins2, loc1=2, loc2=4, fc="none", ec='0.5')


        if i==2 and j==0:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 300, 400, 379, 381
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=1, loc2=3, fc="none", ec='0.5')

            # zoom in lower overlapped curves
            x12, x22, y12, y22 = 900, 1000, 341, 344
            axins2 = inset_axes(axes[j, i], width="20%", height="20%", loc="lower center", bbox_to_anchor=(0, 0.3, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins2.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins2.plot(maximin_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins2.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins2.plot(maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins2.plot(random_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins2.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes[j, i], axins2, loc1=2, loc2=4, fc="none", ec='0.5')

        if i==0 and j==1:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 500, 600, 333, 336
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.2, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=2, loc2=3, fc="none", ec='0.5')

        if i==1 and j==1:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 950, 1050, 375, 378
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=1, loc2=3, fc="none", ec='0.5')

            
        if i==2 and j==1:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 1010, 1050, 376.7, 377.5
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=1, loc2=3, fc="none", ec='0.5')

            
        if i==0 and j==2:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 500, 600, 350.5, 352.5
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=2, loc2=4, fc="none", ec='0.5')

            # zoom in lower overlapped curves
            x12, x22, y12, y22 = 700, 800, 340, 342.5
            axins2 = inset_axes(axes[j, i], width="20%", height="20%", loc="lower center", bbox_to_anchor=(0.1, 0.2, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins2.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins2.plot(maximin_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins2.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins2.plot(maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins2.plot(random_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins2.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes[j, i], axins2, loc1=2, loc2=4, fc="none", ec='0.5')


        if i==1 and j==2:
            # zoom in upper overlapped curves
            x1, x2, y1, y2 = 300, 400, 380, 382.5
            axins = inset_axes(axes[j, i], width="20%", height="20%", loc="center", bbox_to_anchor=(0, 0.15, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[j, i], axins, loc1=1, loc2=3, fc="none", ec='0.5')

            # zoom in lower overlapped curves
            x12, x22, y12, y22 = 100, 160, 343, 345
            axins2 = inset_axes(axes[j, i], width="20%", height="20%", loc="lower center", bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins2.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins2.plot(maximin_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins2.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins2.plot(maximax_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins2.plot(random_mean_growth_rate_mean_discrete)
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins2.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes[j, i].indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes[j, i], axins2, loc1=2, loc2=4, fc="none", ec='0.5')


            

        if i==2 and j==2:
            x1, x2, y1, y2 = 0, 400, 380, 383
            axins = inset_axes(axes[j, i], width="50%", height="50%", loc="center", bbox_to_anchor=(0, 0.1, 1, 1), bbox_transform=axes[j, i].transAxes)
            if "g_maximax" or "all" in policies:
                
                axins.plot(g_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

            if "maximin" or "all" in policies:

                axins.plot(maximin_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


            if "fg_maximax" or "all" in policies:

                axins.plot(fg_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "f_maximax" or "all" in policies:
                
                axins.plot(f_maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)


            if "maximax" or "all" in policies:
                
                axins.plot(maximax_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)



            if "random" or "all" in policies:

                axins.plot(random_mean_growth_rate_mean_discrete)
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


            # # plot hlines
            hlines = [temp_expected_rawlsian_zeta, temp_expected_greedy_zeta]
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_ind = [1, 4]
            for l in range(len(hlines)):
                axins.hlines(hlines[l], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[l]])

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes[2, 2].indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes[2, 2], axins, loc1=2, loc2=4, fc="none", ec='0.5')


# add a zoom-in for the constant-constant cell

axes[0, 1].legend(loc='center right', bbox_to_anchor=(0.96, 0.49), fontsize=10)
plt.tight_layout()
plt.savefig(homo_hetero_signature+"mean_utility_growth_rate_compare_diff_mono_unified.pdf", bbox_inches='tight')
print("plot something")