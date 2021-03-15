from temporal_difference import Temporal_Difference
from monte_carlo import Monte_Carlo

EPISODES = 5000
DISCOUNT = .95
EPSILON = 1
EPSILON_DECAY = 0.997

td = Temporal_Difference()
mc = Monte_Carlo()

def set_hyper_params():
    td.set_hyper_params(episodes = EPISODES, discount = DISCOUNT, epsilon = EPSILON, epsilon_decay = EPSILON_DECAY)
    mc.set_hyper_params(episodes = EPISODES, discount = DISCOUNT, epsilon = EPSILON, epsilon_decay = EPSILON_DECAY)

def run_both_methods():
    set_hyper_params()
    td.run()
    # mc.run()
    # collect results
    # plot results

if __name__ == "__main__":
    run_both_methods()

