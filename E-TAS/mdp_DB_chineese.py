import random

__author__ = 'pegah'
import manage_AS
import numpy as np

"This class manage the extracted data from several txt files" \
" (userlist.txt, wslist.txt, rtdata.txt, tpdata.txt ) to model a MDP"

ftype = np.float32

class Abstract_MDP:

    def getTransitionStatesAndProbs(self, action, state):
        '''
        this is depends on the MDP model(if the model )
        :param state:
        :param action:
        :return:
        '''
        raise NotImplementedError('subclasses must override getTransitionStatesAndProbs!')

    def init_distribution(self):
        """
        :return: a function fir defining an initial distribution on states
        """
        raise NotImplementedError('subclasses must override init_distribution!')

    def getStartState(self):
        """
        :return: returns set of starting states. That depends on our proposed model for MDP
        """
        raise NotImplementedError('subclasses must override getStartState!')

class MDP_SEQ(Abstract_MDP):

    def __init__(self, d, _lambda = None):
        self.d = d #by default for us d = 2
        self.t = manage_AS.manage_AS()
        #self.states = self.getStates()

        # converts _lambda to a np vector so as to use Lambda.dot()
        self.Lambda = np.zeros(d, dtype=ftype)
        if _lambda is not None:
            self.Lambda[:] = _lambda

    def fixStates(self):
        tempo = self.getStates()
        self.states = self.clean_states(tempo)#[-10:]


    def getStates(self):
        """
        recieves the set of sattes for the chinese DB
        :return: list of states with tehir IDs
        """
        # state Id is its index in AS_list
        states_list = self.t.get_AS_list()
        AS_list = states_list
        #AS_list = [i for i in states_list if self.t.get_CS(i) != []]


        #output before shuffle
        #[u'AS4983', u'AS1659', u'AS18515', u'AS7472', u'AS1653', u'AS156', u'AS559', u'AS3661', u'AS553', u'AS2852',
        # u'AS1797', u'AS1955', u'AS3268', u'AS17', u'AS14', u'AS7377', u'AS18', u'AS6192', u'AS17932', u'AS8970', u'AS2107',
        #  u'AS2900', u'AS18047', u'AS7939', u'AS22925', u'AS224', u'AS2496', u'AS3450', u'AS2200', u'AS26', u'AS27', u'AS25',
        #  u'AS6510', u'AS3512', u'AS5786', u'AS29825', u'AS680', u'AS2501', u'AS2500', u'AS36859', u'AS46357', u'AS9112',
        #  u'AS1930', u'AS137', u'AS239', u'AS131', u'AS1938', u'AS7660', u'AS237', u'AS32157', u'AS6356', u'AS4713', u'AS7212',
        #  u'AS786', u'AS31', u'AS33', u'AS32', u'AS7896', u'AS14325', u'AS1741', u'AS1249', u'AS20162', u'AS209', u'AS5661',
        #  u'AS3323', u'AS5723', u'AS11399', u'AS8522', u'AS6431', u'AS6106', u'AS4385', u'AS46', u'AS1111', u'AS1110',
        #  u'AS292', u'AS2012', u'AS1213', u'AS217', u'AS16889', u'AS23329', u'AS12816', u'AS111', u'AS1916', u'AS4730',
        #  u'AS59', u'AS693', u'AS52', u'AS5408', u'AS55', u'AS3', u'AS4', u'AS9', u'AS8', u'AS776', u'AS88', u'AS12925',
        #  u'AS264', u'AS11318', u'AS4201', u'AS13371', u'AS3112', u'AS103', u'AS4538', u'AS15318', u'AS5617', u'AS3388',
        #  u'AS160', u'AS38', u'AS378', u'AS7132', u'AS19262', u'AS18176', u'AS766', u'AS760', u'AS2701', u'AS71', u'AS73',
        #  u'AS20130', u'AS5739', u'AS2552', u'AS3390', u'AS22950', u'AS12464', u'AS2497', u'AS13041', u'AS12093', u'AS16462',
        #  u'AS16461', u'AS8365', u'AS1781', u'AS17716', u'AS7018', u'AS36375', u'AS10881', u'AS22742', u'AS87', u'AS10965']

        #random.shuffle(AS_list, random.random)
        AS_list.append('ASterminal')
        return AS_list

    #I remove the sates without any service id w.r.t the information about throughput and response time
    def clean_states(self, State_list):

        states = []
        for s in State_list:
            if self.t.get_CS(s) != []:
                states.append(s)
        states.append('ASterminal')

        return states

    def getPossibleActions(self, state):
        """
        :param state: for the given state
        :return: returns list of all possible actions in sthe given state
        here actions are service IDs given in file WS-AS.xlsx
        """
        #TODO I may transform all xlsx files to txt files. The txt files are easier and faster to manage
        CS_list = self.t.get_CS(state)
        return CS_list

    def getQoS(self, state, action, time_step, actions_list, matrix, user_id):
        """
        :param state: receives an state name
        :param action: the action ID (web service ID)
        :param time_step: the time step (our model is a time dependent MOMDP)
        :return: returns back set of QoS r
        """

        #actions_list = self.t.get_CS(state)

        int_action = int(action)
        if str(action) in actions_list:
            #tempo = self.t.get_qos(int_action, int(time_step), matrix)
            #return (action, tempo[0], tempo[1])

            if user_id != None:

                output = [[np.float32( matrix[user_id][0][int(time_step)][int_action]),
                           np.float32( 1.0 + matrix[user_id][1][int(time_step)][int_action])]]
                return (action, output)
            else:
                user_id = 0
                output = [[np.float32( matrix[user_id][0][int(time_step)][int_action]),
                           np.float32( 1.0 + matrix[user_id][1][int(time_step)][int_action])]]

                for user_id in range(145)[1:]:
                    trouput = np.float32( matrix[user_id][0][int(time_step)][int_action])
                    response_time = np.float32( 1.0 + matrix[user_id][1][int(time_step)][int_action])
                    output.append([trouput,response_time])

                return (action, output)

        return ()

    def isTerminal(self, state):
        """
        :return: returns set of terminal states. That depends on our proposed model for MDP
        """
        # the terminal state is an numm state defined as ASTerminal
        return state == "ASTerminal"

    def set_Lambda(self,l):
        self.Lambda = np.array(l, dtype=ftype)

    def get_lambda(self):
        return self.Lambda

    def getTransitionStatesAndProbs(self, state, action):
        """
        :param state: current state
        :param action: selected action
        :return: the probability of going to state_ by being un state state and choosing action
        """
        #TODO rewrite this function with try exception

        act = str(action)
        output = []

        nextstate = self.states[self.states.index(state)+1]

        for st in self.states:
            if st == nextstate:
                if act in self.getPossibleActions(state):
                    output.append((st, 1.0))
                else:
                    output.append((st, 0.0))
            else:
                output.append((st, 0.0))
        return output

    def init_distribution(self):
        return

    def getStartState(self):
        return self.states[0]

class MDP_PAR(Abstract_MDP):

    def __init__(self, d, _lambda = None):
        self.d = d #by default for us d = 2
        self.t = manage_AS.manage_AS()
        #self.states = self.getStates()

        # converts _lambda to a np vector so as to use Lambda.dot()
        self.Lambda = np.zeros(d, dtype=ftype)
        if _lambda is not None:
            self.Lambda[:] = _lambda

    def fixStates(self):
        self.states = self.getStates()

    def getStates(self):
        """
        recieves the set of sattes for the chinese DB
        :return: list of states with tehir IDs
        """
        # state Id is its index in AS_list
        states_list = self.t.get_AS_list()
        AS_list = [i for i in states_list if self.t.get_CS(i) != []]

        #random.shuffle(AS_list, random.random)
        AS_list.append('ASterminal')
        return AS_list

    def getPossibleActions(self, state):
        """
        :param state: for the given state
        :return: returns list of all possible actions in sthe given state
        here actions are service IDs given in file WS-AS.xlsx
        """
        #TODO I may transform all xlsx files to txt files. The txt files are easier and faster to manage
        CS_list = self.t.get_CS(state)
        return CS_list

    def getQoS(self, state, action, time_step, actions_list, matrix):
        """
        :param state: receives an state name
        :param action: the action ID (web service ID)
        :param time_step: the time step (our model is a time dependent MOMDP)
        :return: returns back set of QoS r
        """

        #actions_list = self.t.get_CS(state)

        int_action = int(action)
        if str(action) in actions_list:
            #tempo = self.t.get_qos(int_action, int(time_step), matrix)
            #return (action, tempo[0], tempo[1])

            user_id = 0
            output = [[np.float32( matrix[user_id][0][int(time_step)][int_action]),
                       np.float32( matrix[user_id][1][int(time_step)][int_action])]]

            for user_id in range(145)[1:]:
                trouput = np.float32( matrix[user_id][0][int(time_step)][int_action])
                response_time = np.float32( matrix[user_id][1][int(time_step)][int_action])
                output.append([trouput, response_time])

            return (action, output)

        return ()

    def isTerminal(self, state):
        """
        :return: returns set of terminal states. That depends on our proposed model for MDP
        """
        # the terminal state is an numm state defined as ASTerminal
        return state == "ASterminal"

    def set_Lambda(self,l):
        self.Lambda = np.array(l, dtype=ftype)

    def get_lambda(self):
        return self.Lambda

    def getTransitionStatesAndProbs(self, state, action):
        """
        :param state: current state
        :param action: selected action
        :return: the probability of going to state_ by being un state state and choosing action
        """
        #TODO rewrite this function with try exception

        act = str(action)
        output = []

        nextstate = self.states[self.states.index(state)+1]

        if act in self.getPossibleActions(state):
            output.append((nextstate, 1.0))
        else:
            output.append((nextstate, 0.0))
        return output

    def init_distribution(self):
        return

    def getStartState(self):
        return self.states[0]

