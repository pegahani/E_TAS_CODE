import util
import time
from numpy import linalg as LA

__author__ = 'pegah'

import numpy as np
ftype = np.float32


class exact:

    #****************************
    def extract_information(self, valuse):
        values_output = {}
        for key, value in valuse.iteritems():
            values_output[key] = np.float32(valuse[key][1][0])

        return  values_output
    #****************************

    def __init__(self, _mdp, _states_list, _lambda, _discount= 1.0, _height= 63):
        self.mdp = _mdp
        self.d = self.mdp.d
        self.discount = _discount
        self.height = _height

        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.states = _states_list
        self.values = util.Counter(1) # A Counter is a dict with default [0, 0, .., 0]

        self.query_counter_ = 0

    # the exact value iteration goes here

    def computeQValueFromValues_(self, state, action, reward):

        value = [0.0]
        value_real_rt = [0.0]
        transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)

        for nextState, probability in transitionFunction:
            rewards = (np.array(reward)).dot(self.Lambda)
            rewards_real_rt = (np.array([reward[0]*6726.8335, (1.0 - reward[1])*20.0 ])).dot(self.Lambda)

            value += probability * (np.array(rewards) + (self.discount * np.array(self.values[nextState][0])))
            if len(self.values[nextState])>1:
                value_real_rt += probability * (np.array(rewards_real_rt) + (self.discount * np.array(self.values[nextState][1])))
            else:
                value_real_rt += probability * (np.array(rewards_real_rt) + (self.discount * np.array(self.values[nextState][0])))

        return (value, value_real_rt)

    def exact_value_iteraion(self, matrix, user_id, it):

        query_iteration = []
        qos_iteration = []

        result = open("result_exact" + str(it) +".txt", "w")

        states_list = self.states
        optimal_policy = {i:None for i in states_list}
        _time = self.height
        _time_initiale = _time

        while _time > -1:

            print >> result, 'tour', _time
            result.flush()

            valuesCopy = self.values.copy()
            for state in (states_list):

                if _time == _time_initiale:
                    _V_best = self.values[state] #np.zeros(d, dtype=ftype)
                    _V_best_real_rt = self.values[state]
                else:
                    if state != 'ASterminal':
                        _V_best = self.values[state][0]
                        _V_best_real_rt = self.values[state][1]
                    else:
                        _V_best = self.values[state]
                        _V_best_real_rt = self.values[state]

                possible_actions = self.mdp.getPossibleActions(state)

                for action in possible_actions:

                    tempo = self.mdp.getQoS(state, action, _time,possible_actions, matrix, user_id)
                    rewards_list = tempo[1]

                    #rewards_list = tempo[0:2]
                    #matrix = tempo[2]

                    for reward_value in rewards_list:
                        tempo = self.computeQValueFromValues_(state, action, reward_value)
                        Q_d = tempo[0]

                        if Q_d[0] > _V_best[0]:
                            _V_best = Q_d
                            optimal_policy[state] = action
                            _V_best_real_rt = tempo[1]

                        valuesCopy[state] = [_V_best, _V_best_real_rt]

            query_iteration.append(self.query_counter_)

            self.values = valuesCopy
            #print 'self.values', self.values
            _time -= 1

        print >> result, '************'
        print >> result, 'optimal policy', optimal_policy
        print >> result, "final vector value", self.extract_information(self.values)
        print >> result, 'final values with initial distribution', \
            sum([(1.0/len(self.states))*np.array(value) for value in self.values.itervalues()])

        print >> result, "************"
        print >> result, "queries in time" , query_iteration
        result.flush()

        result.close()
        return self.values