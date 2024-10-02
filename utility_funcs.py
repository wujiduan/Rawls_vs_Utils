
import numpy as np
import matplotlib.pyplot as plt
import copy





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

