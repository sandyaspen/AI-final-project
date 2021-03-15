import argparse
import numpy as np
from temporal_difference import Temporal_Difference
from monte_carlo import Monte_Carlo

parser = argparse.ArgumentParser(description='Argument Processor')
parser.add_argument("--tune", type=bool, default=False, help="Defaults False, whether to tune the hyperparameters")


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



def run_both_methods():
    td = Temporal_Difference()
    mc = Monte_Carlo()
    set_hyper_params()
    td.run()
    # mc.run()
    # collect results
    # plot results

if __name__ == "__main__":
    args = parser.parse_args()
    if args.tune:
        tune_hyperparameters()
    #run_both_methods()

