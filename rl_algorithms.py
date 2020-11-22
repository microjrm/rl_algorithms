#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import time
import numpy as np
import mdptoolbox as mdp
import mdptoolbox.example as example
import gym
np.random.seed(23525)

PATH = os.getcwd()

# In[103]:


class BasicLearner:
    def __init__(self, env, n_states, n_actions, n_episodes, gamma, random=False):
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_episodes = n_episodes
        self.random = random
        self.env.reset()
        self.gamma = gamma
    
    def one_step_lookahead(self, state, V):
        action_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            for probability, next_state, reward, _ in self.env.P[state][action]:
                action_values[action] += probability * (reward + 
                                                      (self.gamma * V[next_state]))
        return action_values
    
    def update_policy(self, V, policy):
        for state in range(self.n_states):
            action_values = self.one_step_lookahead(state, V)
            policy[state] = np.argmax(action_values)
        return policy
            
    def value_iteration(self, max_iteration=10000):
        V = np.zeros(self.n_states)
        n_iter = 0
        for i in range(max_iteration):
            prev_v = np.copy(V)
            for state in range(self.n_states):
                action_values = self.one_step_lookahead(state, prev_v)
                best_action_value = np.max(action_values)
                V[state] = best_action_value
                  
            if i % 25 == 0:
                if (np.all(np.isclose(V, prev_v))):
                    n_iter = i
                    break
                    
        optimal_policy = np.zeros(self.n_states, dtype='int8')
        optimal_policy = self.update_policy(V, optimal_policy)
        if n_iter == 0:
            n_iter = max_iteration
                
        return V, optimal_policy, n_iter
    
    def policy_eval(self, V, policy):
        policy_value = np.zeros(self.n_states)
        for state, action in enumerate(policy):
            for probability, next_state, reward, _ in self.env.P[state][action]:
                policy_value[state] += probability * (reward + 
                                                      (self.gamma * V[next_state]))
        return policy_value
            
    def policy_iteration(self, max_iteration=10000):
        V = np.zeros(self.n_states)
        n_iter = 0
        policy = np.random.randint(0, self.n_actions, self.n_states)
        policy_prev = np.copy(policy)
        
        for i in range(max_iteration):
            V = self.policy_eval(V, policy)
            policy = self.update_policy(V, policy)
            
            if i % 25 == 0:
                if (np.all(np.equal(policy, policy_prev))):
                    n_iter = i
                    break
                policy_prev = np.copy(policy)
        if n_iter == 0:
            n_iter = max_iteration
                
        return V, policy, n_iter


class QLearner:
    def __init__(self, alpha, gamma, epsilon, seed=0, verbose=False):
        np.random.seed(seed)
        self.env = gym.make("Taxi-v3").env.unwrapped
        self.states = self.env.observation_space.n
        self.actions = self.env.action_space.n
        self.Q = np.zeros((self.states, self.actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = verbose

    def take_action(self, state, train=True):
        best_action = np.argmax(self.Q[state, :])
        p = np.random.random()
        if p <= self.epsilon:
            a = self.env.action_space.sample()
        else:
            a = best_action
        if train:
            self.epsilon *= self.epsilon
        return a

    def learn(self, n_episodes):
        n_iter = n_episodes
        rewards = np.zeros(n_episodes)
        for i in range(n_episodes):
            reward = 0
            s = self.env.reset()
            if self.verbose:
                print(f'Episode {i}...')
                self.env.render()
            done = False
            previous = self.Q
            while not done:
                a = self.take_action(s, train=False)
                s_, r, done, info = self.env.step(a)
                if self.verbose:
                    self.env.render()
                self.Q[s, a] += self.alpha * (r + self.gamma *
                                              self.Q[s_, int(np.argmax(self.Q[s_, :]))] -
                                              self.Q[s, a])
                s = s_
                reward += r
            rewards[i] = reward

        return rewards

# ## MDP 1: 
# #### Generate a MDP example based on a simple forest management scenario
# ##### A forest is managed by two actions: ‘Wait’ and ‘Cut’. An action is decided each year with first the objective to maintain an old forest for wildlife and second to make money selling cut wood. Each year there is a probability p that a fire burns the forest.

# In[94]:

def learning_experiments():
    policy_iteration_times = np.zeros((1000, 10))
    n_iterations = np.zeros((1000, 10))
    for i, gamma in enumerate(np.linspace(0.1, 0.99, 10)):
        for states in range(2, 1000):
            P, R = example.forest(S=states)
            pi = mdp.mdp.PolicyIteration(P, R, gamma, max_iter=10000)
            pi.run()
            policy_iteration_times[states, i] = pi.time
            n_iterations[states, i] = pi.iter

    np.save(f'{PATH}/policy_iteration_times_forest.npy', policy_iteration_times)
    np.save(f'{PATH}/policy_iteration_n_iter_forest.npy', n_iterations)


    # In[96]:


    value_iteration_times = np.zeros((1000, 10, 10))
    n_iterations = np.zeros((1000, 10, 10))
    for j, epsilon in enumerate(np.linspace(0.1, 0.99, 10)):
        for i, gamma in enumerate(np.linspace(0.1, 0.99, 10)):
            for states in range(2, 1000):
                P, R = example.forest(S=states)
                pi = mdp.mdp.ValueIteration(P, R, discount=gamma, max_iter=10000, epsilon=epsilon)
                pi.run()
                value_iteration_times[states, i, j] = pi.time
                n_iterations[states, i, j] = pi.iter

    np.save(f'{PATH}/value_iteration_times_forest.npy', value_iteration_times)
    np.save(f'{PATH}/value_iteration_n_iter_forest.npy', n_iterations)


    # In[108]:


    Q_iteration_times = np.zeros((1000, 10))
    n_iterations = np.zeros((1000, 10))
    for i, gamma in enumerate(np.linspace(0.1, 0.99, 10)):
        for states in range(2, 1000):
            P, R = example.forest(S=states)
            pi = mdp.mdp.QLearning(P, R, discount=gamma, n_iter=10000)
            pi.run()
            Q_iteration_times[states, i] = pi.time
            n_iterations[states, i] = pi.mean_discrepancy

    np.save(f'{PATH}/Q_iteration_times_forest.npy', Q_iteration_times)
    np.save(f'{PATH}/Q_iteration_n_iter_forest.npy', n_iterations)


    # ## MDP 2: FrozenLake

    # In[98]:
    # In[109]:


    from gym.envs.toy_text.frozen_lake import generate_random_map

    Q_iteration_times = np.zeros((100, 10, 10))
    Q_rewards = np.zeros((100, 10, 10))

    value_n_iterations = np.zeros((100, 10, 10))
    policy_n_iterations = np.zeros((100, 10, 10))
    total_states = np.zeros(100)
    for size in range(2, 100, 5):
        for i, gamma in enumerate(np.linspace(0, 1, 10)):
            for j, epsilon in enumerate(np.linspace(0, 1, 10)):
                random_map = generate_random_map(size=size, p=0.8)
                environment = gym.make('FrozenLake-v0', desc=random_map)
                test = QLearner(0.1, gamma, epsilon, verbose=False)
                start = time.time()
                n = test.learn(50)
                Q_iteration_times[size, i, j] = time.time() - start
                Q_rewards[size, i, j] = n[-1]

    np.save(f'{PATH}/Q_iteration_times_grid.npy', Q_iteration_times)
    np.save(f'{PATH}/Q_iteration_rewards_grid.npy', Q_rewards)


    # In[106]:


    value_iteration_times = np.zeros((100, 10))
    policy_iteration_times = np.zeros((100, 10))

    value_n_iterations = np.zeros((100, 10))
    policy_n_iterations = np.zeros((100, 10))
    total_states = np.zeros(100)
    for size in range(2, 100, 5):
        for i, gamma in enumerate(np.linspace(0, 1, 10)):
            random_map = generate_random_map(size=size, p=0.8)
            environment = gym.make('FrozenLake-v0', desc=random_map)
            total_states[size] = environment.nS
            agent = BasicLearner(environment, environment.nS,
                                 environment.nA, 5000, gamma)
            start = time.time()
            opt_v2, opt_policy2, value_iter = agent.value_iteration()
            value_iteration_times[size, i] = time.time() - start
            value_n_iterations[size, i] = value_iter

            start = time.time()
            opt_v2, opt_policy2, policy_iter = agent.policy_iteration()
            policy_iteration_times[size, i] = time.time() - start
            policy_n_iterations[size, i] = policy_iter

    np.save(f'{PATH}/num_states_grid.npy', total_states)
    np.save(f'{PATH}/policy_iteration_times_grid.npy', policy_iteration_times)
    np.save(f'{PATH}/value_iteration_times_grid.npy', value_iteration_times)
    np.save(f'{PATH}/value_iteration_n_iter_grid.npy', value_n_iterations)
    np.save(f'{PATH}/policy_iteration_n_iter_grid.npy', policy_n_iterations)


# In[ ]:




