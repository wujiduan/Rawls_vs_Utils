import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 
import copy
import os
import multiprocessing
import random
from math import comb
from utility_funcs import compute_gini, read_data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import heapq




initial_value = read_data('read_sipp/sipp_2014_wave_1.csv')

budget_num = 10
agent_num = len(initial_value)
print("agent num:", agent_num)
homo_decay_ub = 60
homo_decay_lb = 10
# f^- > (N-1)*g^+ ensures the positivity
# the coefficient 1/budget_num model the fact that the less the total resource is, the less resource one individual will get.
# homo_return_lb = homo_decay_ub * agent_num * 4 / budget_num
# homo_return_ub = homo_return_lb + (homo_decay_lb+homo_decay_ub) * agent_num / (2*budget_num)
homo_return_lb = homo_decay_ub * agent_num * 4 
homo_return_ub = homo_return_lb + (homo_decay_lb+homo_decay_ub) * agent_num / 2 
print("return upperbound:", homo_return_ub)
print("return lowerbound:", homo_return_lb)

# expected zeta for Rawlsian policy
homo_expected_zeta = (budget_num * homo_return_ub - (agent_num-budget_num) * homo_decay_lb) / agent_num
print(homo_expected_zeta)
homo_expected_zeta_greedy = (budget_num * homo_return_ub - (agent_num-budget_num) * homo_decay_ub) / agent_num
print("expected greedy rbar:", homo_expected_zeta_greedy)

# ================define paths======================

param_folder = "sipp_linear_final_params"
if not os.path.exists(param_folder):
    os.makedirs(param_folder)
shape_params_path = param_folder + '/sipp_linear_shape_params_multi_budget' + str(budget_num) + '.txt'
results_folder = "multi_budget_results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
signature = results_folder + "/homo_hetero_sipp_linear_budget" + str(budget_num) + "_"



# ===================generate shape params===========================
try: 
    # 'a' mode - to append
    # 'w' mode - generate a new set of parameters everytime.
    # 'x' mode - create and write in a file if it does not exist, otherwise do nothing
    with open(shape_params_path, 'w') as file:
        
        counter = 0
        while counter < agent_num:

            # we fix two end points and interpolate two linear functions.
            # as long as return_ub-return_lb > decay_ub-decay_lb
            return_left_turning_x = np.random.randint(-2100, -2000)
            decay_left_turning_x = np.random.randint(-1900, -1850)
            return_right_turning_x = np.random.randint(400, 800)
            decay_right_turning_x = np.random.randint(-1600, -1500)
            temp_return_alpha = (homo_return_ub - homo_return_lb) / (return_right_turning_x - return_left_turning_x)
            temp_return_beta = homo_return_lb - return_left_turning_x * temp_return_alpha
            
            temp_decay_alpha = (homo_decay_lb - homo_decay_ub) / (decay_right_turning_x - decay_left_turning_x)
            temp_decay_beta = homo_decay_ub - decay_left_turning_x * temp_decay_alpha

            
            if temp_return_alpha > -temp_decay_alpha:
                # make the unsigned decay decreasing w.r.t. x
                file.write(str(temp_return_alpha) + ","+ str(temp_return_beta) +","+ str(temp_decay_alpha) +","+ str(temp_decay_beta) + "\n")
                counter += 1
                print(counter)
                # break
            
except FileExistsError:
    print("Shape parameters already generated!")

# ===========================read shape params==============================

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

print(return_alphas)
print(len(return_alphas))

# ===================set other params=========================
return_ubs = np.array([homo_return_ub for _ in range(agent_num)])
return_lbs = np.array([homo_return_lb for _ in range(agent_num)])
decay_ubs = np.array([homo_decay_ub for _ in range(agent_num)])
decay_lbs = np.array([homo_decay_lb for _ in range(agent_num)])


max_time_horizon = 10000
sample_num = 100



# possible policies: "maximin", "maximax", "fg_maximax", "f_maximax", "random"
policies = ["all"]
# for recording gini indices
if_gini = False
# for testing different tie-breaking rules for max-g policy
if_differ_tie_break = False

'''  
We define the framework for comparison, and use return/decay functions as input parameters. 
This framework defined the return & decay is varying w.r.t. the utility, and they all include some symmetric noises.
The codes are vectorized w.r.t. agent_num.
'''
def run_comparison_fast(sample_ind):


    np.random.seed(sample_ind)
    if "maximin" or "all" in policies:
        maximin_agent_utilities = np.zeros(agent_num)
        maximin_mean_utilities = np.zeros(max_time_horizon)
        maximin_mean_growth_rate = np.zeros(max_time_horizon-1)
        if if_gini:
            maximin_gini = np.zeros(max_time_horizon)


    if "fg_maximax" or "all" in policies:
        fg_maximax_agent_utilities = np.zeros(agent_num)
        fg_maximax_mean_utilities = np.zeros(max_time_horizon)
        fg_maximax_mean_growth_rate = np.zeros(max_time_horizon-1)
        if if_gini:
            fg_maximax_gini = np.zeros(max_time_horizon)


    if "f_maximax" or "all" in policies:
            
        f_maximax_agent_utilities = np.zeros(agent_num)
        f_maximax_mean_utilities = np.zeros(max_time_horizon)
        f_maximax_mean_growth_rate = np.zeros(max_time_horizon-1)
        if if_gini:
            f_maximax_gini = np.zeros(max_time_horizon)

    if "g_maximax" or "all" in policies:
            
        g_maximax_agent_utilities = np.zeros(agent_num)
        g_maximax_mean_utilities = np.zeros(max_time_horizon)
        g_maximax_mean_growth_rate = np.zeros(max_time_horizon-1)
        if if_gini:
            g_maximax_gini = np.zeros(max_time_horizon)


    if "maximax" or "all" in policies:
        maximax_agent_utilities = np.zeros(agent_num)
        maximax_mean_utilities = np.zeros(max_time_horizon)
        maximax_mean_growth_rate = np.zeros(max_time_horizon-1)
        if if_gini:
            maximax_gini = np.zeros(max_time_horizon)


    if "random" or "all" in policies:
        random_agent_utilities = np.zeros(agent_num)
        # mean utility at each time step
        random_mean_utilities = np.zeros(max_time_horizon)
        random_mean_growth_rate = np.zeros(max_time_horizon-1)
        # record gini coefficient
        if if_gini:
            random_gini = np.zeros(max_time_horizon)

    # variance bound
    variance_bound = budget_num * (max(return_ubs) + max(decay_ubs))
    # noise range
    noise_bound = 2 * budget_num * (max(return_ubs) + max(decay_ubs))
    
    
    maximin_agent_utilities = initial_value.copy()
    fg_maximax_agent_utilities = initial_value.copy()
    f_maximax_agent_utilities = initial_value.copy()
    # for max-g policy, we use a slightly different tie-breaker and hence need to sort the values
    if not if_differ_tie_break:
        g_maximax_agent_utilities = sorted(initial_value.copy())
    else:
        g_maximax_agent_utilities = initial_value.copy()
    maximax_agent_utilities = initial_value.copy()
    random_agent_utilities = initial_value.copy()


    maximin_mean_utilities[0] = np.mean(maximin_agent_utilities)
    if if_gini:
        _, maximin_gini[0] = compute_gini(maximin_agent_utilities, 1/(2*agent_num))
    

    maximax_mean_utilities[0] = np.mean(maximax_agent_utilities) 
    if if_gini:
        _, maximax_gini[0] = compute_gini(maximax_agent_utilities, 1/(2*agent_num))
    

    fg_maximax_mean_utilities[0] = np.mean(fg_maximax_agent_utilities)
    if if_gini:
        _, fg_maximax_gini[0] = compute_gini(fg_maximax_agent_utilities, 1/(2*agent_num))
    

    f_maximax_mean_utilities[0] = np.mean(f_maximax_agent_utilities)
    if if_gini:
        _, f_maximax_gini[0] = compute_gini(f_maximax_agent_utilities, 1/(2*agent_num))
    
    g_maximax_mean_utilities[0] = np.mean(g_maximax_agent_utilities)
    if if_gini:
        _, g_maximax_gini[0] = compute_gini(g_maximax_agent_utilities, 1/(2*agent_num))
    

    random_mean_utilities[0] = np.mean(random_agent_utilities)
    if if_gini:
        _, random_gini[0] = compute_gini(random_agent_utilities, 1/(2*agent_num))
    
    # now we compare two policies: putting out fires and greedy one 
    for t in range(max_time_horizon-1):
        
        # shared symmetric noises in [-(return_ub+decay_ub), +(return_ub+decay_ub)]
        # comment the following line to use different noises for each policies
        # temp_noise = np.round(np.clip(np.random.normal(0, np.sqrt(noise_bound), agent_num), -2*noise_bound, 2*noise_bound))

        if "maximin" or "all" in policies:
        # "putting out fires" policy
        # compute the expected intervention return and the shock decays
        
            min_indices = [idx for idx, _ in heapq.nlargest(budget_num, enumerate(maximin_agent_utilities), key=lambda x: (-x[1], -x[0]))]

            maximin_decay_indicator = np.ones(agent_num)
            for ind in min_indices: 
                maximin_decay_indicator[ind] = 0
            temp_maximin_return = np.round(np.clip(return_alphas*maximin_agent_utilities+return_betas, return_lbs, return_ubs))
            maximin_agent_utilities += temp_maximin_return * (1-maximin_decay_indicator)
            temp_maximin_decay = np.round(np.clip(decay_alphas*maximin_agent_utilities+decay_betas, decay_lbs, decay_ubs))
            maximin_agent_utilities -= temp_maximin_decay * maximin_decay_indicator
            # add noises
            maximin_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))
                
            maximin_mean_utilities[t+1] = np.mean(maximin_agent_utilities)
            maximin_mean_growth_rate[t] = (maximin_mean_utilities[t+1]-maximin_mean_utilities[0]) / (t+1)
            if if_gini:
                _, maximin_gini[t+1] = compute_gini(maximin_agent_utilities, 1/(2*agent_num))
            
        if "maximax" or "all" in policies:
            # maximax policy
            # add noises
            
            max_indices = [idx for idx, _ in heapq.nlargest(budget_num, enumerate(maximax_agent_utilities), key=lambda x: (x[1], -x[0]))]

            maximax_decay_indicator = np.ones(agent_num)
            for ind in max_indices:
                maximax_decay_indicator[ind] = 0
            temp_maximax_return = np.round(np.clip(return_alphas * maximax_agent_utilities + return_betas, return_lbs, return_ubs))
            maximax_agent_utilities += temp_maximax_return * (1-maximax_decay_indicator)
            temp_maximax_decay = np.round(np.clip(decay_alphas*maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
            maximax_agent_utilities -= temp_maximax_decay * maximax_decay_indicator
            maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))

            maximax_mean_utilities[t+1] = np.mean(maximax_agent_utilities)
            maximax_mean_growth_rate[t] = (maximax_mean_utilities[t+1]-maximax_mean_utilities[0]) / (t+1)
            if if_gini:
                _, maximax_gini[t+1] = compute_gini(maximax_agent_utilities, 1/(2*agent_num))
            
        if "fg_maximax" or "all" in policies:
            # fg_maximax policy
            temp_fg_return = np.round(np.clip(return_alphas*fg_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
            temp_fg_decay = np.round(np.clip(decay_alphas*fg_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
            temp_fplusg = temp_fg_return + temp_fg_decay
            max_fg_indices = [idx for idx, _ in heapq.nlargest(budget_num, enumerate(temp_fplusg), key=lambda x: (x[1], -x[0]))]

        
            fg_maximax_decay_indicator = np.ones(agent_num)
            for ind in max_fg_indices:
                fg_maximax_decay_indicator[ind] = 0
            fg_maximax_agent_utilities -= temp_fg_decay * fg_maximax_decay_indicator
            fg_maximax_agent_utilities += temp_fg_return * (1-fg_maximax_decay_indicator)
            # add noises
            fg_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))

            fg_maximax_mean_utilities[t+1] = np.mean(fg_maximax_agent_utilities)
            fg_maximax_mean_growth_rate[t] = (fg_maximax_mean_utilities[t+1]-fg_maximax_mean_utilities[0]) / (t+1)
            if if_gini:
                _, fg_maximax_gini[t+1] = compute_gini(fg_maximax_agent_utilities, 1/(2*agent_num))
            

        if "f_maximax" or "all" in policies:
            # f_maximax policy
            temp_f_return = np.round(np.clip(return_alphas*f_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
            temp_f_decay = np.round(np.clip(decay_alphas*f_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))

            max_f_indices = [idx for idx, _ in heapq.nlargest(budget_num, enumerate(temp_f_return), key=lambda x: (x[1], -x[0]))]

            f_maximax_decay_indicator = np.ones(agent_num)
            for ind in max_f_indices:
                f_maximax_decay_indicator[ind] = 0
            f_maximax_agent_utilities -= temp_f_decay * f_maximax_decay_indicator
            f_maximax_agent_utilities += temp_f_return * (1-f_maximax_decay_indicator)
            # add noises
            f_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))

            f_maximax_mean_utilities[t+1] = np.mean(f_maximax_agent_utilities)
            f_maximax_mean_growth_rate[t] = (f_maximax_mean_utilities[t+1]-f_maximax_mean_utilities[0]) / (t+1)
            if if_gini:
                _, f_maximax_gini[t+1] = compute_gini(f_maximax_agent_utilities, 1/(2*agent_num))
            

        if "g_maximax" or "all" in policies:
            # g_maximax policy
            temp_g_return = np.round(np.clip(return_alphas*g_maximax_agent_utilities+return_betas, return_lbs, return_ubs))
            temp_g_decay = np.round(np.clip(decay_alphas*g_maximax_agent_utilities+decay_betas, decay_lbs, decay_ubs))
            # since welfare values have been sorted in ascending order, returning the smallest index is enough for obtaining
            # the individual with the highest g with smallest welfare level
            
            max_g_indices = [idx for idx, _ in heapq.nlargest(budget_num, enumerate(temp_g_decay), key=lambda x: (x[1], -x[0]))]
            
            g_maximax_decay_indicator = np.ones(agent_num)
            for ind in max_g_indices:
                g_maximax_decay_indicator[ind] = 0
            g_maximax_agent_utilities -= temp_g_decay * g_maximax_decay_indicator
            g_maximax_agent_utilities += temp_g_return * (1-g_maximax_decay_indicator)
            # add noises
            
            g_maximax_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))
            if not if_differ_tie_break:
                g_maximax_agent_utilities = sorted(g_maximax_agent_utilities.copy())

            g_maximax_mean_utilities[t+1] = np.mean(g_maximax_agent_utilities)
            g_maximax_mean_growth_rate[t] = (g_maximax_mean_utilities[t+1]-g_maximax_mean_utilities[0]) / (t+1)
            if if_gini:
                _, g_maximax_gini[t+1] = compute_gini(g_maximax_agent_utilities, 1/(2*agent_num))
            


        if "random" or "all" in policies:
            # random policy
            temp_random_decay = np.round(np.clip(decay_alphas*random_agent_utilities+decay_betas, decay_lbs, decay_ubs))
            temp_rand_indices = np.random.choice(np.arange(0, agent_num), size=budget_num, replace=False)
            random_decay_indicator = np.ones(agent_num)
            for ind in temp_rand_indices:
                random_decay_indicator[ind] = 0
            random_agent_utilities -= temp_random_decay * random_decay_indicator
            temp_random_return = np.round(np.clip(return_alphas * random_agent_utilities + return_betas, return_lbs, return_ubs))
            random_agent_utilities += temp_random_return * (1-random_decay_indicator)
            # add noises
            random_agent_utilities += np.round(np.clip(np.random.normal(0, np.sqrt(variance_bound), agent_num), -noise_bound, noise_bound))
                        
            random_mean_utilities[t+1] = np.mean(random_agent_utilities)
            random_mean_growth_rate[t] = (random_mean_utilities[t+1]-random_mean_utilities[0]) / (t+1)
            if if_gini:
                _, random_gini[t+1] = compute_gini(random_agent_utilities, 1/(2*agent_num))
                
        # We take the measure of utilitarianism: average utility, \sum_{i=1}^N Ui(t)/(N*t)
    if "maximin" or "all" in policies:

        with open(signature+"maximin_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(maximin_mean_growth_rate, file)
        

        if if_gini:
            
            with open(signature+"maximin_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(maximin_gini, file)
            

    if "maximax" or "all" in policies:
        
        with open(signature+"maximax_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(maximax_mean_growth_rate, file)
        
        if if_gini:
            
            with open(signature+"maximax_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(maximax_gini, file)
            

    if "fg_maximax" or "all" in policies:

        
        with open(signature+"fg_maximax_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(fg_maximax_mean_growth_rate, file)
        

        if if_gini:
            
            with open(signature+"fg_maximax_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(fg_maximax_gini, file)
            
    if "f_maximax" or "all" in policies:
        
        with open(signature+"f_maximax_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(f_maximax_mean_growth_rate, file)
        

        if if_gini:
            with open(signature+"f_maximax_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(f_maximax_gini, file)
            
    if "g_maximax" or "all" in policies:
        
        with open(signature+"g_maximax_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(g_maximax_mean_growth_rate, file)
        

        if if_gini:
            
            with open(signature+"g_maximax_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(g_maximax_gini, file)
            
        

    if "random" or "all" in policies:
        
        
        with open(signature+"random_mean_growth_rate_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
            pickle.dump(random_mean_growth_rate, file)
        
        if if_gini:
            
            with open(signature+"random_gini_multi_budget_" + str(sample_ind)+ ".pk", 'wb') as file:
                pickle.dump(random_gini, file)
            
    return 0






if __name__ == "__main__":

    params = []
    for i in range(sample_num):
        params.append(i)

    if_only_adjust_plot = False
    if not if_only_adjust_plot:
        pool_size = multiprocessing.cpu_count()
        with multiprocessing.Pool(pool_size) as pool:
            res = pool.map(run_comparison_fast, params)

    fig, axes = plt.subplots()
    axes.set_xlabel("Time step t", fontsize=18)
    axes.set_ylabel("Social welfare", fontsize=18)
    

    xs = np.array([i for i in range(max_time_horizon-1)])
    
    if not if_differ_tie_break:
        if budget_num != 100:
            axins = inset_axes(axes, width="15%", height="15%", loc = "lower left", bbox_to_anchor=(0.4, 0.35, 1, 1), bbox_transform=axes.transAxes)
            axins2 = inset_axes(axes, width="15%", height="15%", loc = "center left", bbox_to_anchor=(0.3, 0.15, 1, 1), bbox_transform=axes.transAxes)


    if "g_maximax" or "all" in policies:
        g_maximax_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"g_maximax_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                g_maximax_mean_growth_rate_samples[:,ind] = pickle.load(file)

        g_maximax_mean_growth_rate_mean_discrete = np.mean(g_maximax_mean_growth_rate_samples, axis=1)
        g_maximax_mean_growth_rate_std_discrete = np.std(g_maximax_mean_growth_rate_samples, axis=1)
        
        axes.plot(g_maximax_mean_growth_rate_mean_discrete, label = "max-g (Rawlsian)")
        axes.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(g_maximax_mean_growth_rate_mean_discrete, label = "max-g (Rawlsian)")
                axins.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(g_maximax_mean_growth_rate_mean_discrete, label = "max-g (Rawlsian)")
                axins2.fill_between(xs, g_maximax_mean_growth_rate_mean_discrete-g_maximax_mean_growth_rate_std_discrete, g_maximax_mean_growth_rate_mean_discrete+g_maximax_mean_growth_rate_std_discrete, alpha=0.4)


    if "maximin" or "all" in policies:


        maximin_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"maximin_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                maximin_mean_growth_rate_samples[:,ind] = pickle.load(file)
        
        maximin_mean_growth_rate_mean_discrete = np.mean(maximin_mean_growth_rate_samples, axis=1)
        maximin_mean_growth_rate_std_discrete = np.std(maximin_mean_growth_rate_samples, axis=1)
        
        axes.plot(maximin_mean_growth_rate_mean_discrete, label="min-U (Rawlsian)")
        axes.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)
        
        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(maximin_mean_growth_rate_mean_discrete, label="min-U (Rawlsian)")
                axins.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(maximin_mean_growth_rate_mean_discrete, label="min-U (Rawlsian)")
                axins2.fill_between(xs, maximin_mean_growth_rate_mean_discrete-maximin_mean_growth_rate_std_discrete, maximin_mean_growth_rate_mean_discrete+maximin_mean_growth_rate_std_discrete, alpha=0.4)


    if "fg_maximax" or "all" in policies:

        fg_maximax_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"fg_maximax_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                fg_maximax_mean_growth_rate_samples[:,ind] = pickle.load(file)
        
        fg_maximax_mean_growth_rate_mean_discrete = np.mean(fg_maximax_mean_growth_rate_samples, axis=1)
        fg_maximax_mean_growth_rate_std_discrete = np.std(fg_maximax_mean_growth_rate_samples, axis=1)
        
        axes.plot(fg_maximax_mean_growth_rate_mean_discrete, label="max-fg (utilitarian)")
        axes.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)

        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(fg_maximax_mean_growth_rate_mean_discrete, label="max-fg (utilitarian)")
                axins.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(fg_maximax_mean_growth_rate_mean_discrete, label="max-fg (utilitarian)")
                axins2.fill_between(xs, fg_maximax_mean_growth_rate_mean_discrete-fg_maximax_mean_growth_rate_std_discrete, fg_maximax_mean_growth_rate_mean_discrete+fg_maximax_mean_growth_rate_std_discrete, alpha=0.4)


    if "f_maximax" or "all" in policies:
        f_maximax_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"f_maximax_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                f_maximax_mean_growth_rate_samples[:,ind] = pickle.load(file)
        
        f_maximax_mean_growth_rate_mean_discrete = np.mean(f_maximax_mean_growth_rate_samples, axis=1)
        f_maximax_mean_growth_rate_std_discrete = np.std(f_maximax_mean_growth_rate_samples, axis=1)
        
        axes.plot(f_maximax_mean_growth_rate_mean_discrete, label = "max-f (utilitarian)")
        axes.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)

        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(f_maximax_mean_growth_rate_mean_discrete, label = "max-f (utilitarian)")
                axins.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(f_maximax_mean_growth_rate_mean_discrete, label = "max-f (utilitarian)")
                axins2.fill_between(xs, f_maximax_mean_growth_rate_mean_discrete-f_maximax_mean_growth_rate_std_discrete, f_maximax_mean_growth_rate_mean_discrete+f_maximax_mean_growth_rate_std_discrete, alpha=0.4)



    
    if "maximax" or "all" in policies:
        maximax_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"maximax_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                maximax_mean_growth_rate_samples[:,ind] = pickle.load(file)
        
        maximax_mean_growth_rate_mean_discrete = np.mean(maximax_mean_growth_rate_samples, axis=1)
        maximax_mean_growth_rate_std_discrete = np.std(maximax_mean_growth_rate_samples, axis=1)
        
        axes.plot(maximax_mean_growth_rate_mean_discrete, label="max-U (utilitarian)")
        axes.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)

        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(maximax_mean_growth_rate_mean_discrete, label="max-U (utilitarian)")
                axins.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(maximax_mean_growth_rate_mean_discrete, label="max-U (utilitarian)")
                axins2.fill_between(xs, maximax_mean_growth_rate_mean_discrete-maximax_mean_growth_rate_std_discrete, maximax_mean_growth_rate_mean_discrete+maximax_mean_growth_rate_std_discrete, alpha=0.4)


    

    if "random" or "all" in policies:
        
        random_mean_growth_rate_samples = np.zeros((max_time_horizon-1, sample_num))
        for ind in range(sample_num):
            
            with open(signature+"random_mean_growth_rate_multi_budget_" + str(ind) + ".pk", 'rb') as file:
                random_mean_growth_rate_samples[:,ind] = pickle.load(file)
        
        random_mean_growth_rate_mean_discrete = np.mean(random_mean_growth_rate_samples, axis=1)
        random_mean_growth_rate_std_discrete = np.std(random_mean_growth_rate_samples, axis=1)
        
        axes.plot(random_mean_growth_rate_mean_discrete, label="Random")
        axes.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


        if not if_differ_tie_break:
            if budget_num != 100:
                axins.plot(random_mean_growth_rate_mean_discrete, label="Random")
                axins.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)

                axins2.plot(random_mean_growth_rate_mean_discrete, label="Random")
                axins2.fill_between(xs, random_mean_growth_rate_mean_discrete-random_mean_growth_rate_std_discrete, random_mean_growth_rate_mean_discrete+random_mean_growth_rate_std_discrete, alpha=0.4)


    # plot hlines
    hlines = [homo_expected_zeta, homo_expected_zeta_greedy]
    hline_labels = [r'$\bar{R}_{min-U}$ (Rawlsian)', r'$\bar{R}_{max-U}$ (utilitarian)']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_ind = [1, 4]
    for i in range(len(hlines)):
        axes.hlines(hlines[i], xmin=0, xmax=max_time_horizon, linestyle = '--', colors=color_cycle[color_ind[i]], label=hline_labels[i])

    if budget_num == 100:
        axes.set_xlim(-300, 10000)
    elif budget_num == 10:
        axes.set_xlim(-300, 10000)
    if budget_num == 100:
        axes.set_ylim(27440, 27495)
    elif budget_num == 10:
        axes.set_ylim(2685, 2745)
    else:
        pass
    axes.legend(loc='center right', bbox_to_anchor=(1.5, 0.7), fontsize=12)
    axes.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    

    if budget_num == 1:
        # the coordinates are hard-coded to improve the illustration
        x1, x2, y1, y2 = 950, 1050, 225, 227
        x12, x22, y12, y22 = 950, 1050, 259, 261
    elif budget_num == 10:
        x1, x2, y1, y2 = 1950, 2050, 2704, 2708
        x12, x22, y12, y22 = 1950, 2050, 2727, 2731
    else:
        print("Please adjust the zoom-in window accordingly!")
        x1, x2, y1, y2 = 950, 1050, 234, 237
        x12, x22, y12, y22 = 950, 1050, 265, 267

    if not if_differ_tie_break:
        
        if budget_num != 100:
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            axes.indicate_inset_zoom(axins, edgecolor='black')
            mark_inset(axes, axins, loc1=2, loc2=4, fc="none", ec='0.5')

            axins2.set_xlim(x12, x22)
            axins2.set_ylim(y12, y22)
            axins2.set_xticks([])
            axins2.set_yticks([])
            axes.indicate_inset_zoom(axins2, edgecolor='black')
            mark_inset(axes, axins2, loc1=1, loc2=3, fc="none", ec='0.5')
    
    fig_folder = "multi_budget_figs"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_signature = fig_folder + "/homo_hetero_sipp_linear_budget" + str(budget_num) + "_"
    plt.savefig(fig_signature+"mean_utility_growth_rate_compare_discrete_multi_budget" + str(budget_num) + ".pdf", bbox_inches='tight')
    plt.show()