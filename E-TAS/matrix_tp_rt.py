import time
import subprocess
import operator

__author__ = 'pegah'

import  numpy as np
ftype = np.float64

class make_matrix:
    def __init__(self, which_file):

        self.which_file = "./Files/" + which_file

        return

    def file_len(self, fname):
        p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE)
        result, err = p.communicate()
        if p.returncode != 0:
            raise IOError(err)
        return int(result.strip().split()[0])

    def get_martix(self):

        count = 0
        output = np.zeros((self.file_len(self.which_file), 5), ftype)
        #output = np.zeros((20, 5), ftype)
        #file = "tp-rt-merge.txt"
        file = self.which_file
        #start = time.time()
        with open(file) as qos:
            for line in qos:
                words_list = line.split()
                output[count,:] = words_list
                count += 1

        ind = np.lexsort(((output[:, 1]).astype(int), (-output[:, 2]).astype(int)))
        output = output[ind]

        output[:,[0, 1, 2]] = output[:,[0, 1, 2]].astype(int)

        return output

    def get_matrix_response_time(self):
        """
        generate a matrix of 64 rows (time ids) and 4500 columns (service ids)
        :return: each element represents the RESPONSE TIME value for a service id at special time id
        """

        output = np.zeros((64, 4500), ftype)
        file = self.which_file
        with open(file) as qos:
            for line in qos:
                words_list = line.split()
                output[int(words_list[2]),int(words_list[1])] = words_list[4]

        return output

    def get_matrix_throughput(self):
        """
        generate a matrix of 64 rows (time ids) and 4500 columns (service ids)
        :return: each element represents the THROUGHPOUT value for a service id at special time id
        """

        output = np.zeros((64, 4500), ftype)
        file = self.which_file
        with open(file) as qos:
            for line in qos:
                words_list = line.split()
                output[int(words_list[2]), int(words_list[1])] = words_list[3]

        return output

    # def structure_mdp(self, matrix):
    #     dic = {}
    #     for time in xrange(63,-1,-1):
    #         for state in (self.states):
    #             possible_actions = self.mdp.getPossibleActions(state)
    #             for action in possible_actions:
    #                 tempo = self.mdp.getQoS(state, action, time, matrix)
    #                 rewards_list = tempo[0:2]
    #                 matrix = tempo[2]
    #                 dic[(state, action, time)] = rewards_list
    #
    #     return dic

    def get_all_users_matrix(self):

        output = dict.fromkeys(range(145))
        for key in output:
            # first matrix represents throughput values while the second matrix represents response time.
            output[key] = [np.zeros((64, 4500), ftype), np.zeros((64, 4500), ftype)]

        file = self.which_file

        with open(file) as qos:
            for line in qos:
                words_list = line.split()
                # fill throughput part
                output[int(words_list[0])][0][int(words_list[2]), int(words_list[1])] = words_list[3]
                # fill response time part
                output[int(words_list[0])][1][int(words_list[2]), int(words_list[1])] = words_list[4]

        return output