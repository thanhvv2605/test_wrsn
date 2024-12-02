import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.random.RandomController import RandomController
from rl_env.WRSN import WRSN


def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print(net.env.now)
        yield net.env.timeout(1.0)


network = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               , agent_type_path="physical_env/mc/mc_types/default.yaml"
               , num_agent=1)
action_dim = len(network.net.listChargingLocations) + 1
controller = RandomController(action_dim)

request = network.reset()
for id, _ in enumerate(network.net.targets_active):
    if _ == 0:
        print(id)

while not request["terminal"]:
    print(request["agent_id"], request["action"], request["terminal"])
    action = controller.select_action()
    # print("embedding of all node in network: ", network.get_state(0))
    # print("embedding of all node in network: ", network.get_enegy())

    request = network.step(request["agent_id"], action)
    print(network.net.env.now)

print(network.net.env.now)