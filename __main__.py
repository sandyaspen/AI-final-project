import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from temporal_difference import Temporal_Difference
from monte_carlo import Monte_Carlo

parser = argparse.ArgumentParser(description='Argument Processor')
parser.add_argument("--tune", type=bool, default=False, help="Defaults False, whether to tune the hyperparameters")
cpu_count = mp.cpu_count()
#Don't use them all
if cpu_count < 3: cpu_count = 1


TEST_COUNT = 5 


def tune_hyperparameters():
    potential_discounts = [(100 - i)/100 for i in range(11)]
    print(potential_discounts)
    results = {"mc": {"discounts": []}, "td": {"discounts": []}}
    for discount in potential_discounts:
        td_mean_rewards = []
        mc_mean_rewards = []
        for _ in range(100):
            td = Temporal_Difference()
            td.set_hyper_params(discount=discount)
            q, _ = td.train()
            _, episode_rewards = td.test(q=q, episodes=100)
            td_mean_rewards.append(np.mean(episode_rewards))
            mc = Monte_Carlo()
            mc.set_hyper_params(discount=discount)
            q, qq, _ = mc.train()
            _, _, episode_rewards = mc.test(q=q, qq=qq, episodes=100)
            mc_mean_rewards.append(np.mean(episode_rewards))
        results["mc"]["discounts"].append(np.mean(td_mean_rewards))
        results["td"]["discounts"].append(np.mean(mc_mean_rewards))
    best_discount = {"discount": 0, "total_mean": 0} 
    for i, (mc_mean, td_mean) in enumerate(zip(results["mc"]["discounts"], results["td"]["discounts"])):
        if mc_mean + td_mean > best_discount["total_mean"]:
            best_discount["total_mean"] = mc_mean + td_mean
            best_discount["discount"] = potential_discounts[i]
    print(results)
    print(best_discount)


    """
    td = Temporal_Difference()
    q, _ = td.train()
    _, episode_rewards = td.test(q=q, episodes=100)
    print(episode_rewards)
    print(np.mean(episode_rewards))

    mc = Monte_Carlo()
    q, qq, _ = mc.train()
    _, _, episode_rewards = mc.test(q=q, qq=qq, episodes=100)
    print(episode_rewards)
    print(np.mean(episode_rewards))
    """



def full_test_monte_carlo():
    episode_counts = list(range(500,5001,500))
    
    print("Running monte carlo test...")
    #Start multiprocessing to speed up tests
    with mp.Pool(processes=cpu_count) as pool:
        results = pool.map(test_mc_5_times, episode_counts)
    #unpack results
    episodes, avg_rewards, stddev_rewards = zip(*results)    

    #Plot
    fig, ax = plt.subplots()
    ax.errorbar(episodes, avg_rewards, yerr=stddev_rewards)
    ax.set(xlabel='Number of Test Episodes',  ylabel='Average Rewards',
        title="Monte Carlo Test Results")
    ax.grid()
    fig.savefig("MCresults.png")

def test_mc_5_times(ep_count):
    rewards = []
    stddev = []
    for test_num in range(TEST_COUNT):
        mc = Monte_Carlo()
        mc.train(ep_count)
        _, _, test_rewards = mc.test(None, 500)
        rewards.append(np.average(test_rewards))
        stddev.append(np.std(test_rewards))
    #Remove high and low trials
    min_idx = rewards.index(min(rewards))
    del rewards[min_idx]
    del stddev[min_idx]
    max_idx = rewards.index(max(rewards))
    del rewards[max_idx]
    del stddev[max_idx]
    #Add average, stddev of these three to running list
    return ep_count, np.average(rewards), np.average(stddev)



def full_test_temporal_differnce():
    episode_counts = list(range(500,5001,500))
    
    print("Running temporal difference test...")
    #Start multiprocessing to speed up tests
    with mp.Pool(processes=cpu_count) as pool:
        results = pool.map(test_td_5_times, episode_counts)
    #unpack results
    episodes, avg_rewards, stddev_rewards = zip(*results)    

    #Plot
    fig, ax = plt.subplots()
    ax.errorbar(episodes, avg_rewards, yerr=stddev_rewards)
    ax.set(xlabel='Number of Test Episodes',  ylabel='Average Rewards',
        title="Temporal Difference Test Results")
    ax.grid()
    fig.savefig("TDresults.png")

def test_td_5_times(ep_count):
    rewards = []
    stddev = []
    for test_num in range(TEST_COUNT):
        td = Temporal_Difference()
        td.train(ep_count)
        _, test_rewards = td.test(None, 500)
        rewards.append(np.average(test_rewards))
        stddev.append(np.std(test_rewards))
    #Remove high and low trials
    min_idx = rewards.index(min(rewards))
    del rewards[min_idx]
    del stddev[min_idx]
    max_idx = rewards.index(max(rewards))
    del rewards[max_idx]
    del stddev[max_idx]
    #Add average, stddev of these three to running list
    return ep_count, np.average(rewards), np.average(stddev)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.tune:
        tune_hyperparameters()
    full_test_temporal_differnce()
    full_test_monte_carlo()

