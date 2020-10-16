'''
Description:
    The goal of this assignment is to implement three basic algorithms to solve multi-armed bandit problem.
        1. Epislon-Greedy Alogorithm 
        2. Upper-Confidence-Bound Action Selection
        3. Gradient Bandit Algorithms
    Follow the instructions in code to complete your assignment :)
'''
# import standard libraries
import random
import argparse
import numpy as np

# import other
from env import Gaussian_MAB, Bernoulli_MAB
from algo import EpislonGreedy, UCB, Gradient
from utils import plot

# function map
FUNCTION_MAP = {'e-Greedy': EpislonGreedy, 
                'UCB': UCB,
                'grad': Gradient}
 
# train function 
def train(args, env, algo):
    reward = np.zeros(args.max_timestep)
    if algo == UCB:
        parameter = args.c
    else:
        parameter = args.epislon

    # start multiple experiments
    for _ in range(args.num_exp):
        # start with new environment and policy
        mab = env(args.num_of_bandits)
        agent = algo(args.num_of_bandits, parameter)
        for t in range(args.max_timestep):
            # choose action first
            a = agent.act(t)

            # get reward from env
            r = mab.step(a)

            # update
            agent.update(a, r)

            # append to result
            reward[t] += r

        if _%10 ==0 :
            print('exp: ', _)
    
    avg_reward = reward / args.num_exp
    return avg_reward

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", "--num_of_bandits", type=int, 
                        default=10, help="number of bandits")
    parser.add_argument("-algo", "--algo",
                        default="e-Greedy", choices=FUNCTION_MAP.keys(),
                        help="Algorithm to use")
    parser.add_argument("-epislon", "--epislon", type=float,
                        default=0.0, help="epislon for epislon-greedy algorithm")
    parser.add_argument("-c", "--c", type=float,
                        default=1, help="c for UCB")
    parser.add_argument("-max_timestep", "--max_timestep", type=int,
                        default=500, help="Episode")
    parser.add_argument("-num_exp", "--num_exp", type=int,
                        default=100, help="Total experiments to run")
    parser.add_argument("-plot", "--plot", action='store_true',
                        help='plot the results')
    parser.add_argument("-runAll", "--runAll", action='store_true',
                        help='run all three algos')
    parser.add_argument("-vary_e", "--vary_e", action='store_true',
                        help='')
    parser.add_argument("-vary_c", "--vary_c", action='store_true',
                        help='')
    parser.add_argument("-vary_bd", "--vary_bd", action='store_true',
                        help='')
    args = parser.parse_args()

    # start training
    import time

    tStart = time.time()
    avg_reward = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])
    tEnd = time.time()
    print(tEnd-tStart)
    if args.plot:
        plot(np.expand_dims(avg_reward, axis=0), [args.algo])



    ##############################################################################
    # After you implement all the method, uncomment this part, and then you can  #  
    # use the flag: --runAll to show all the results in a single figure.         #
    ##############################################################################

    if args.runAll:
        _all = ['e-Greedy', 'UCB', 'grad']
        avg_reward = np.zeros([len(_all), args.max_timestep])
        for algo in _all:
            idx = _all.index(algo)
            avg_reward[idx] = train(args, Gaussian_MAB, FUNCTION_MAP[algo])
        plot(avg_reward, _all)

    if args.vary_e:
        e_list = [ 0.05, 0.1, 0.15, 0.20]
        avg_reward = np.zeros([len(e_list), args.max_timestep])
        for i, e in enumerate(e_list):
            args.epislon = e
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])
        arg = FUNCTION_MAP[args.algo]
        plot_list = ['e=0.05','e=0.1','e=0.15','e=0.20']
        plot(avg_reward, plot_list)

        print('hello world')

    if args.vary_c:
        c_list = [ 1,2,3,4,5,6]
        avg_reward = np.zeros([len(c_list), args.max_timestep])
        for i, v in enumerate(c_list):
            args.c = v
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP['UCB'])
        plot_list = ['c=1','c=2','c=3','c=4','c=5','c=6']
        plot(avg_reward, plot_list)

    if args.vary_bd:
        import time
        b_list = [10, 50, 250]
        avg_reward = np.zeros([len(b_list), args.max_timestep])
        time_list=[]
        for i, v in enumerate(b_list):
            print(f'bd number{v}')
            args.num_of_bandits = v
            tStart = time.time()
            avg_reward[i] = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])#UCB e-Greedy grad
            tEnd = time.time()
            time_list.append(tStart-tEnd)
        print(time_list)
        plot_list = ['10', '50', '250']
        plot(avg_reward, plot_list)




