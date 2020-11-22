#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

PATH = os.getcwd()
def plot():
    # In[27]:
    # Policy iteration
    pi = np.load(f'{PATH}/policy_iteration_times_forest.npy')
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        plt.plot(pi[:, i], label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Policy Iteration of Forest Management MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[86]:


    # Policy iteration
    pi = np.load(f'{PATH}/policy_iteration_n_iter_forest.npy')
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        plt.plot(pi[:, i], label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Policy Iteration of Forest Management MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Number of iterations to convergence')
    plt.show()


    # In[55]:


    # Policy iteration
    pi = np.load(f'{PATH}/policy_iteration_times_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    print(num_states)
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Policy Iteration of FrozenLake MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[95]:


    # Policy iteration
    pi = np.load(f'{PATH}/policy_iteration_n_iter_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Policy Iteration of FrozenLake MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Number of iterations to convergence')
    plt.show()


    # In[96]:


    # Value iteration
    pi = np.load(f'{PATH}/value_iteration_n_iter_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Value Iteration of FrozenLake MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Number of iterations to convergence')
    plt.show()


    # In[72]:


    # Value iteration
    pi = np.load(f'{PATH}/value_iteration_times_forest.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    print(num_states)
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, -1]
        temp = temp[temp != 0]
        plt.plot(temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Value Iteration of Forest Management MDP ($\epsilon$=0.99)')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[73]:


    # Value iteration
    pi = np.load(f'{PATH}/value_iteration_times_forest.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    print(num_states)
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, 0]
        temp = temp[temp != 0]
        plt.plot(temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Value Iteration of Forest Management MDP ($\epsilon$=0.1)')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[88]:


    pi = np.load(f'{PATH}/value_iteration_n_iter_forest.npy')
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        plt.plot(pi[:, i, -1], label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Value Iteration of Forest Management MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Number of iterations to convergence')
    plt.show()


    # In[67]:


    # Value iteration
    pi = np.load(f'{PATH}/value_iteration_times_forest.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    print(num_states)
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, -1, i]
        temp = temp[temp != 0]
        plt.plot(temp, label=f'$\epsilon$={gamma}')
    plt.legend()
    plt.title('Value Iteration of Forest MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[70]:


    # Value iteration
    pi = np.load(f'{PATH}/value_iteration_times_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    print(num_states)
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Value Iteration of FrozenLake MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[30]:


    # Q-learning
    pi = np.load(f'{PATH}/Q_iteration_times_forest.npy')
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        plt.plot(pi[2:, i], label=f'$\gamma$={gamma}')
    plt.legend()
    plt.title('Q-learning of Forest Management MDP')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[109]:


    # Q-learning
    pi = np.load(f'{PATH}/Q_iteration_times_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, -1]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Q-learning of FrozenLake MDP $\epsilon=0.99$')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[110]:


    # Q-learning
    pi = np.load(f'{PATH}/Q_iteration_times_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, 0]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Q-learning of FrozenLake MDP $\epsilon=0.1$')
    plt.xlabel('Number of States')
    plt.ylabel('Convergence Time (sec)')
    plt.show()


    # In[120]:


    # Q-learning
    pi = np.load(f'{PATH}/Q_iteration_rewards_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, 0]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Q-learning of FrozenLake MDP $\epsilon=0.1$')
    plt.xlabel('Number of States')
    plt.ylabel('Average reward across episodes')
    plt.show()


    # In[121]:


    # Q-learning
    pi = np.load(f'{PATH}/Q_iteration_rewards_grid.npy')
    num_states = np.load('num_states_grid.npy')
    num_states = num_states[num_states != 0]
    figure = plt.figure(figsize=(10,5))
    for i, gamma in enumerate([0.1, 0.198, 0.297, 0.396, 0.495,
                               0.594, 0.693, 0.792, 0.891, 0.99]):
        temp = pi[:, i, -1]
        temp = temp[temp != 0]
        plt.plot(num_states, temp, label=f'$\gamma$={gamma}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Q-learning of FrozenLake MDP $\epsilon=0.99$')
    plt.xlabel('Number of States')
    plt.ylabel('Average reward across episodes')
    plt.show()


# In[ ]:




