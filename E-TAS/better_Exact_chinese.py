import cplex
import operator
import time
from numpy import linalg as LA

__author__ = 'pegah'

import numpy as np
import util

ftype = np.float32

class better_exact_chinese:
    def __init__(self, _mdp, _states_list, _lambda, _discount= 1.0, _height= 63):

        self.mdp = _mdp
        self.d = self.mdp.d
        self.discount = _discount
        self.height = _height

        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.states = _states_list
        self.values = util.Counter(self.d) # A Counter is a dict with default [0, 0, .., 0]

        # it is a global variable for definig if an action selected for a state in the optimal policy
        self.change_checker = False

        #print [i for i in self.values.iterkeys()]

        self.query_counter_ = 0
        self.Lambda_inequalities = []

    def computeQValueFromValues(self, state, action, reward):

        value = [0.0]*self.mdp.d
        transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)

        for nextState, probability in transitionFunction:
            rewards = reward
            value += probability * ( np.array(rewards) + (self.discount * np.array(self.values[nextState])) )

        return value

    "value iteration for discrete time MDP with a finite horizon"
    def better_exact_iteration(self, matrix):

        #errors_iteration = []
        #qos_iteration = []

        #variables for checking calculation time
        global data_time, step_time, states_time, actions_time, qos_time

        result = open("result_exact_better" + ".txt", "w")

        states_list = self.states
        d = self.d

        #initial optimal policy
        optimal_policy = {i:None for i in states_list}

        "the iteration start here"
        _time = self.height #stop at the bottom of the horizontal length

        while _time > -1:
            print >> result,'tour = ', _time
            result.flush()

            valuesCopy = self.values.copy()

            for state in (states_list):

                _V_best_d = self.values[state] #np.zeros(d, dtype=ftype)
                possible_actions = self.mdp.getPossibleActions(state)

                for action in possible_actions:

                    tempo = self.mdp.getQoS(state, action, _time, possible_actions, matrix)
                    rewards_list = tempo[0:2]
                    matrix = tempo[2]

                    for reward_value in rewards_list[1]:
                        Q_d = self.computeQValueFromValues(state, action, reward_value)

                        if Q_d.dot(self.Lambda) > _V_best_d.dot(self.Lambda):
                            _V_best_d = Q_d
                            optimal_policy[state] = action

                        valuesCopy[state] = _V_best_d


            self.values = valuesCopy
            #print 'self.values', self.values
            _time -= 1

            #print 'errors_iteration', query_iteration
            #temp_vect = sum([(1.0/len(self.states))*np.array(value) for value in self.values.itervalues()])
            #qos_iteration.append(LA.norm( temp_vect, np.inf))
            #print 'qos_iteration', qos_iteration
            #errors_iteration.append(self.values)


        print >> result, '************'
        print >> result, 'optimal policy', optimal_policy
        print >> result, 'final vector value', self.values

        print >> result, '******************'
        #print >> result, 'for computing error', errors_iteration

        result.flush()
        result.close()

        return self.values
