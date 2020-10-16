"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """

        self._Q[action] = ((self._action_N[action]-1)*self._Q[action] + immi_reward)/self._action_N[action]



        ################### Your code here #######################
        #raise NotImplementedError('[EpislonGreedy] update function NOT IMPLEMENTED')

        ##########################################################

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore.
        Return: action selection
        """
        ################### Your code here #######################
        if np.random.rand() > self._epislon:
            max_value = np.max(self._Q)
            idx = np.where(self._Q == max_value)
            idx = np.array(idx).flatten()
            a = np.random.choice(idx)

        else:
            a = np.random.randint(0,self._nb)
        self._action_N[a] += 1
        return a

        #raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        ##########################################################

class UCB(object):
    """
    Implementation of upper confidence bound.
    """
    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        #raise NotImplementedError('[UCB] update function NOT IMPLEMENTED')
        self._Q[action] = ((self._action_N[action] - 1) * self._Q[action] + immi_reward) / self._action_N[action]
        pass
        ##########################################################

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        """
               Step 3: Choose the action via greedy or explore.
               Return: action selection
               """
        ################### Your code here #######################

        idx = np.where(self._action_N == 0)
        idx = np.array(idx).flatten()
        if len(idx) == 0:
            value_list = []
            for i in range(self._nb):
                value = self._Q[i] + self._c * np.sqrt(np.log(t) / self._action_N[i])
                value_list += [value]
                value_np = np.array(value_list).flatten()
            action = np.argmax(value_np)
        else:
            action = np.random.choice(idx)

        self._action_N[action] += 1

        return action

        # raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        ##########################################################
        ################### Your code here #######################
        #raise NotImplementedError('[UCB] act function NOT IMPLEMENTED')
        ##########################################################


class Gradient(object):
    """
    Implementation of your gradient-based method
    """
    def __init__(self, NumofBandits=10, alpha=0.1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._H = np.zeros(self._nb, dtype=float)
        self._H_exp = np.ones(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self.alpha = alpha
        self.r_avg = 0.0
        self.time_step = 0
        self.pi =np.zeros(self._nb, dtype=float)
        for i in range(self._nb):
            self.pi[i] = 1.0 / self._nb

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        ##########################################################
        self.r_avg = (self.r_avg*(self.time_step-1)+immi_reward)/self.time_step
        self._H[action] = self._H[action] + self.alpha*(immi_reward-self.r_avg)*(1-self.pi[action])
        for a in range(self._nb):
            if a != action:
                self._H[a] = self._H[a] - self.alpha * (immi_reward - self.r_avg)*(self.pi[action])
        self._H_exp = np.exp(self._H)



        denominator = 0.0
        for i in range(self._nb):
            denominator += self._H_exp[i]
        for a in range(self._nb):
            numerator = self._H_exp[a]
            self.pi[a] = numerator/denominator
        pass

    def act(self, t):

        action = np.random.choice(self._nb,p=self.pi)
        self.time_step += 1
        return action
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
        ################### Your code here #######################
        # choose action
        ##########################################################

    # def pif(self,action):
    #     denominator = 0
    #     for i in range(self._nb):
    #         denominator += self._H_exp[i]
    #     numerator = self._H_exp[action]
    #     return numerator/denominator

