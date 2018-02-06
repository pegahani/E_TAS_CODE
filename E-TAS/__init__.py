import random
import time
import pickle
import manage_AS
import numpy as np
import mdp_DB_chineese
import ABVI_chineseDB
import Exact_chinese
import matrix_tp_rt
import better_Exact_chinese


#lambdas
# lambda1 [0.319797998295, 0.680202001705]
# lambda2 [0.8573741847324399, 0.14262581526756013]
# lambda3 [0.1696287781131175, 0.8303712218868825]
# lambda4 [0.6451844883834318, 0.3548155116165682]
# lambda5 [0.47245438345, 0.52754561655]

#****************************
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


for it in range(145):

    result = open("result_final_user" + str(it)+ ".txt", "w")
    _lambda = [0.319797998295, 0.680202001705]
    _discount = 1.0

    _user_id = it

    print >> result, 'lambda = ', _lambda

    MDP = mdp_DB_chineese.MDP_SEQ(2)
    #MDP = mdp_DB_chineese.MDP_PAR(2)
    MDP.fixStates()
    states = MDP.states

    print states
    print len(states)

    #i = 1
    #a = matrix_tp_rt.make_matrix("user" + str(i) + ".txt")

    a = matrix_tp_rt.make_matrix("tp-rt-normalize.txt")

    #b_all_users = a.get_all_users_matrix()
    #save_obj(b_all_users,'b_all_users')

    b_all_users = load_obj('b_all_users')
    start1 = time.time()

    #find the algorithm result
    abv = ABVI_chineseDB.abvi_chinese(_mdp= MDP, _states_list = states, _lambda =_lambda, _discount=_discount, _height= 63 )
    values = abv.interactive_value_iteration(b_all_users, user_id = _user_id, it= it)

    print >> result, 'final values ivi ', abv.extract_information(values)
    finish2 = time.time()
    print >> result, 'time sec ivi = ', finish2-start1
    result.flush()

    start = time.time()

    print "********************"

    #find the exact response
    exact = Exact_chinese.exact(_mdp= MDP, _states_list= states, _lambda= _lambda, _discount= _discount, _height= 63)
    exact_values = exact.exact_value_iteraion(b_all_users, user_id = _user_id, it= it)

    print >> result, 'final values exact', exact.extract_information(exact_values)
    finish = time.time()
    print >> result, 'time sec Exact = ', finish-start
    result.flush()

    # start = time.time()
    # exact_better = better_Exact_chinese.better_exact_chinese(_mdp= MDP, _states_list = states, _lambda =_lambda, _discount=1.0, _height= 63 )
    # exact_better_values = exact_better.better_exact_iteration(b)
    # finish = time.time()
    #
    # print >> result, 'final values exact', exact_better_values
    #
    # print >> result, 'time sec Exact = ', finish-start
    # result.flush()

    result.close()
