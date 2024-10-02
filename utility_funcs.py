
import numpy as np
import matplotlib.pyplot as plt
import copy




# this function distribute budget based on the maximin policy
# we use the multiples of the unit delta throughout - assume the inputs are multiples of units.
# we use > 1/2 to determine whether the value is positive
# leximax policy can be programmed using dynamical programming
# welfare should be in array format
# adding parameter return: we generaliza the function to the case where one budget can lead to 
# different increases in the welfare level while the ultimate goal is the same: we try to 
# maximize the lowest welfare, the second lowest welfare with the budget we have.
# NOTE: Here the budget allocation is a one-time thing, the new welfare of an agent should equal to 
# old_welfare + assigned_budget * its return per budget
''' 
In this function, we assume one unit of budget will increase the welfare by one unit.
If the return function is increasing w.r.t. the welfare, then multiply the return with the
increase is the correct answer. The answer still holds.
This function does not consider that there is upperbound to be capped.
'''
def leximax_policy(welfares, budget):


    # determine the positions where welfares are not 0
    agent_num = len(welfares)
    max_welfare = np.max(welfares)
    # minimum value except 0
    if max_welfare < 1/2:
        print("All agents has dropped out the system.")
        return 0
    
    min_welfare = max_welfare 
    for i in range(agent_num):
        if welfares[i] > 1/2 and welfares[i] < min_welfare:
            min_welfare = welfares[i]

    left = min_welfare 
    right = max_welfare + budget 
    mid = np.floor((left + right) / 2)
    increase = np.zeros(agent_num)
    while left <= right and budget > 1/2: 
        
        diff = np.array([mid-welfares[i] if welfares[i] <= mid and welfares[i] >= 1/2 else 0 for i in range(agent_num)])
        if np.sum(diff) <= budget:
            left = mid + 1
            mid = np.floor((left + right) / 2)
            increase = copy.deepcopy(diff)
        else: 
            right = mid - 1
            mid = np.floor((left + right) / 2) 

    # base case 1 is we can distribute budget perfectly, budget = 0 and np.sum(increase) = 0
    # , and base case 2 is that we have budget left but not enough for increasing
    # the minimum welfare, in this case, it must be the number of agents with lowest welfare is larger than the budget 
    if budget - np.sum(increase) > 1/2 and np.sum(increase) < 1/2:
        # choose budget agents to give them one unit of budget

        for i in range(len(welfares)):
            if welfares[i] > 1/2 and welfares[i] == min_welfare and budget > 1/2:
                increase[i] += 1 
                budget -= 1

    # have extra budget for maximizing the second smallest welfares
    elif budget - np.sum(increase) > 1/2 and np.sum(increase) > 1/2:
        
        increase += leximax_policy(welfares+increase, budget-np.sum(increase))

    # print("debug:", increase, "budget:", budget)
    return increase

'''
In this function, it's a more generalized version of the above function leximax_policy
'''
def leximax_policy_general(welfares, returns, budget):
    
    agent_num = welfares.shape[0]
    increases = np.zeros(agent_num)
    increase_budgets = np.zeros(agent_num)
    if budget <= 0:
        return increases, increase_budgets
    
    min_welfare = np.inf
    for i in range(agent_num):
        if welfares[i] < min_welfare and welfares[i] > 0 and returns[i] > 0:
            min_welfare = welfares[i]
            # print("min welfare:", min_welfare)
    
    min_inds_returns = {}
    
    for i in range(agent_num):
        
        if welfares[i] == min_welfare:
            min_inds_returns[i] = returns[i]
    
    
    if len(min_inds_returns) > 0 and budget > 0:
        
        # print("returns:", min_inds_returns, "budget:", budget)
        if len(min_inds_returns) == budget: 
            for key in min_inds_returns:
                increases[key] = returns[key]
                increase_budgets[key] += 1

        elif len(min_inds_returns) > budget:
            
            sorted_inds = sorted(min_inds_returns.items(), key = lambda x:x[1], reverse=True)
            for b in range(budget):
                ind = sorted_inds[b][0]
                increases[ind] += returns[ind]
                increase_budgets[ind] += 1
        else: 
            for key in min_inds_returns:
                increases[key] = returns[key]
                increase_budgets[key] += 1
            # print("welfare + increases:", welfares + increases)
            add_increases, add_increase_budgets = leximax_policy_general(welfares+increases, returns, budget-len(min_inds_returns))
            increases += add_increases
            increase_budgets += add_increase_budgets
    # returning the budget is convenient for plotting
    # print("welfare:", welfares, "increases:", increase_budgets)
    return increases, increase_budgets
    




''' 
greedy policy maximizes the sum of returns without considering the upperbound to be capped.
NOTE: for agents outside the range of threshold, the return should be 0.
'''
def greedy_policy(welfares, returns, budget):
    
    max_return = max(returns)
    max_ids = []
    agent_num = welfares.shape[0]
    for i in range(agent_num):
        if returns[i] == max_return:
            max_ids.append(i)
 
    increases = np.zeros(agent_num)
    increase_budgets = np.zeros(agent_num)

    if max_return > 0:
    # otherwise, all agents are outside of the eligible range
        selected_ids = np.random.choice(max_ids, size=budget)
        for id in selected_ids:

            increases[id] += returns[id] 
            increase_budgets[id] += 1

    return increases, increase_budgets




''' 
The following function computes the utilities given welfares.
Function types can be: 1) increasing, 2) decreasing, 3) concave, 4) convex
more exactly, the 'concave' denotes a first increasing and then decreasing trend.
'concave' denotes a first decreasing and then increasing trend.
NOTE: only agents with welfares between [1, threshold] have a nonzero return. Since we 
consider a threshold instead of an upperbound, the returns are not capped.
'''
def compute_returns(welfares, threshold, return_func):
    
    agent_num = len(welfares)
    returns = np.zeros(agent_num)
    if return_func == 'increasing':
        for i in range(agent_num):
            if welfares[i] > 0 and welfares[i] <= threshold:
                returns[i] =  np.ceil(welfares[i] / (threshold * 0.3))

    elif return_func == 'decreasing':
        for i in range(agent_num):
            if welfares[i] > 0 and welfares[i] <= threshold:
                returns[i] = np.ceil((1-welfares[i]/threshold)/0.3)

    elif return_func == 'concave':
        for i in range(agent_num):
            if welfares[i] > 0 and welfares[i] <= threshold:
                returns[i] = np.ceil((4*welfares[i]/threshold-4*(welfares[i]/threshold)**2)/0.3)
    else:
        for i in range(agent_num):
            if welfares[i] > 0 and welfares[i] <= threshold:
                returns[i] = np.ceil((4*welfares[i]/threshold*(welfares[i]/threshold-1)+1)/0.3)

    return returns

def compute_returns_upperbound(welfares, upperbound, return_func):

    agent_num = len(welfares)
    returns = np.zeros(agent_num)
    if return_func == 'increasing':
        for i in range(agent_num):
            if welfares[i] > 0:
                returns[i] =  min(np.ceil(welfares[i]/(upperbound * 0.3)), upperbound-welfares[i])

    elif return_func == 'decreasing':
        for i in range(agent_num):
            if welfares[i] > 0:
                returns[i] = min(np.ceil((1-welfares[i]/upperbound)/0.3), upperbound-welfares[i])

    elif return_func == 'concave':
        for i in range(agent_num):
            if welfares[i] > 0:
                returns[i] = min(np.ceil((4*welfares[i]/upperbound-4*(welfares[i]/upperbound)**2)/0.3), upperbound-welfares[i])
    else:
        for i in range(agent_num):
            if welfares[i] > 0:
                returns[i] = min(np.ceil((4*welfares[i]/upperbound*(welfares[i]/upperbound-1)+1)/0.3), upperbound-welfares[i])

    return returns

''' 
This function computes the Gini index of the distribution.
'''
def compute_gini(welfares, quantile):
    
    agent_num = welfares.shape[0]
    sorted_welfares = np.sort(welfares)
    welfare_sum = np.sum(welfares)
    bins = np.array([np.sum(sorted_welfares[max(0, int(np.floor(i * agent_num * quantile))): min(agent_num, int(np.floor((i+1) * agent_num * quantile)))])/welfare_sum for i in range(int(1/quantile))])
    accu_bins = np.array([np.sum(bins[:i]) for i in range(1, bins.shape[0]+1)])
    gini_coef = 1- 2 * np.sum(accu_bins) * quantile
    # print(bins)
    # print(sorted_welfares)
    # plt.hist(welfares, bins=agent_num)
    # plt.show()
    return accu_bins, gini_coef

'''
res[t, k] the surviving probability for agent start at time step t and welfare level k. 
with time horizon 
'''
def compute_survived_probs(prob, time_horizon, threshold):

    # # firstly compute the ruin probabilities, and use surviving prob = 1-ruin prob in the end
    # ruin_probs = np.zeros((time_horizon+1, time_horizon+threshold+1))
    # # boundary condition 
    # for t in range(time_horizon+1):
    #     ruin_probs[t, 0] = 1

    
    # for t in range(time_horizon-1, -1, -1):
    #     for k in range(1, threshold+1+t):

    #         ruin_probs[t, k] = prob * ruin_probs[t+1, k+1] + (1 - 2*prob) * ruin_probs[t+1, k] + prob * ruin_probs[t+1,k-1]


    # slightly change the last column, if we change it to another increasing sequence
    # is it possible to have the decreasing theorem?
    # debug_ones = np.ones((time_horizon+1, threshold+2))
    # for l in range(threshold+2):
    #     debug_ones[time_horizon, l] = 1 - (threshold+2-l) * 0.06

    # directly compute surviving probabilities
    survive_probs = np.zeros((time_horizon+1, time_horizon+threshold+1))
    # boundary condition
    for l in range(1, time_horizon+1):
        # correct version
        survive_probs[time_horizon, l] = 1

    for l in range(time_horizon+1, time_horizon+threshold+1):
        # debug version
        survive_probs[time_horizon, l] = 1 - (threshold+threshold+1-l) * 0.03

    for t in range(time_horizon-1, -1, -1):
        for k in range(1, threshold+1+t):

            survive_probs[t, k] = prob * survive_probs[t+1, k+1] + (1 - 2*prob) * survive_probs[t+1, k] + prob * survive_probs[t+1, k-1]


    return survive_probs[:, :threshold+2]